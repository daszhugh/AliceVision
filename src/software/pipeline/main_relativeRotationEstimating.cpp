// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/types.hpp>
#include <aliceVision/config.hpp>

#include <aliceVision/system/Timer.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/cmdline/cmdline.hpp>

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/feature/imageDescriberCommon.hpp>

#include <aliceVision/track/TracksBuilder.hpp>
#include <aliceVision/track/tracksUtils.hpp>
#include <aliceVision/track/trackIO.hpp>

#include <aliceVision/camera/Pinhole.hpp>

#include <aliceVision/robustEstimation/ACRansac.hpp>
#include <aliceVision/multiview/RelativePoseKernel.hpp>
#include <aliceVision/multiview/relativePose/Rotation3PSolver.hpp>

#include <aliceVision/sfm/liealgebra.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <aliceVision/sfm/pipeline/relativePoses.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>

#include <cstdlib>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

using namespace aliceVision;

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace bg = boost::geometry;


bool robustRotation(Mat3& R, std::vector<size_t>& vecInliers, 
                    const Mat& x1, const Mat& x2, 
                    std::mt19937& randomNumberGenerator,
                    const size_t maxIterationCount, const size_t minInliers)
{
    using KernelType = multiview::RelativePoseSphericalKernel<
                        multiview::relativePose::Rotation3PSolver,
                        multiview::relativePose::RotationError,
                        robustEstimation::Mat3Model
                    >;

    KernelType kernel(x1, x2);

    robustEstimation::Mat3Model model;
    vecInliers.clear();

    // robustly estimation of the Essential matrix and its precision
    robustEstimation::ACRANSAC(kernel, randomNumberGenerator, vecInliers, 1024, &model, std::numeric_limits<double>::infinity());

    if(vecInliers.size() < minInliers)
    {
        return false;
    }

    R = model.getMatrix();

    return true;
}

void computeCovisibility(std::map<Pair, unsigned int>& covisibility, const track::TracksMap& mapTracks)
{
    for(const auto& item : mapTracks)
    {
        const auto& track = item.second;

        for(auto it = track.featPerView.begin(); it != track.featPerView.end(); it++)
        {
            Pair p;
            p.first = it->first;

            for(auto next = std::next(it); next != track.featPerView.end(); next++)
            {
                p.second = next->first;

                if(covisibility.find(p) == covisibility.end())
                {
                    covisibility[p] = 0;
                }
                else
                {
                    covisibility[p]++;
                }
            }
        }
    }
}

int aliceVision_main(int argc, char** argv)
{
    // command-line parameters
    std::string sfmDataFilename;
    std::vector<std::string> featuresFolders;
    std::string tracksFilename;
    std::string outputDirectory;
    int rangeStart = -1;
    int rangeSize = 1;
    const size_t minInliers = 35;

    // user optional parameters
    std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);


    int randomSeed = std::mt19937::default_seed;

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
    ("input,i", po::value<std::string>(&sfmDataFilename)->required(), "SfMData file.")
    ("tracksFilename,i", po::value<std::string>(&tracksFilename)->required(), "Tracks file.")
    ("output,o", po::value<std::string>(&outputDirectory)->required(), "Path to the output directory.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
    ("featuresFolders,f", po::value<std::vector<std::string>>(&featuresFolders)->multitoken(), "Path to folder(s) containing the extracted features.")
    ("describerTypes,d", po::value<std::string>(&describerTypesName)->default_value(describerTypesName),feature::EImageDescriberType_informations().c_str())
    ("rangeStart", po::value<int>(&rangeStart)->default_value(rangeStart), "Range image index start.")
    ("rangeSize", po::value<int>(&rangeSize)->default_value(rangeSize), "Range size.");

    CmdLine cmdline("AliceVision relativeRotationEstimating");

    cmdline.add(requiredParams);
    cmdline.add(optionalParams);
    if(!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    // set maxThreads
    HardwareContext hwc = cmdline.getHardwareContext();
    omp_set_num_threads(hwc.getMaxThreads());
    
    std::mt19937 randomNumberGenerator(randomSeed);

    // load input SfMData scene
    sfmData::SfMData sfmData;
    if(!sfmDataIO::Load(sfmData, sfmDataFilename, sfmDataIO::ESfMData::ALL))
    {
        ALICEVISION_LOG_ERROR("The input SfMData file '" + sfmDataFilename + "' cannot be read.");
        return EXIT_FAILURE;
    }

    // Define range to compute
    if(rangeStart != -1)
    {
      if(rangeStart < 0 || rangeSize < 0 || rangeStart > sfmData.getViews().size())
      {
        ALICEVISION_LOG_ERROR("Range is incorrect");
        return EXIT_FAILURE;
      }

      if(rangeStart + rangeSize > sfmData.getViews().size())
      {
        rangeSize = sfmData.getViews().size() - rangeStart;
      }
    }
    else
    {
        rangeStart = 0;
        rangeSize = sfmData.getViews().size();
    }
    ALICEVISION_LOG_DEBUG("Range to compute: rangeStart=" << rangeStart << ", rangeSize=" << rangeSize);


    // get imageDescriber type
    const std::vector<feature::EImageDescriberType> describerTypes =
        feature::EImageDescriberType_stringToEnums(describerTypesName);
        

    // features reading
    feature::FeaturesPerView featuresPerView;
    ALICEVISION_LOG_INFO("Load features");
    if(!sfm::loadFeaturesPerView(featuresPerView, sfmData, featuresFolders, describerTypes))
    {
        ALICEVISION_LOG_ERROR("Invalid features.");
        return EXIT_FAILURE;
    }

    // Load tracks
    ALICEVISION_LOG_INFO("Load tracks");
    std::ifstream tracksFile(tracksFilename);
    if(tracksFile.is_open() == false)
    {
        ALICEVISION_LOG_ERROR("The input tracks file '" + tracksFilename + "' cannot be read.");
        return EXIT_FAILURE;
    }
    std::stringstream buffer;
    buffer << tracksFile.rdbuf();
    boost::json::value jv = boost::json::parse(buffer.str());
    track::TracksMap mapTracks(track::flat_map_value_to<track::Track>(jv));

    // Compute tracks per view
    ALICEVISION_LOG_INFO("Estimate tracks per view");
    track::TracksPerView mapTracksPerView;
    for(const auto& viewIt : sfmData.views)
    {
        // create an entry in the map
        mapTracksPerView[viewIt.first];
    }
    track::computeTracksPerView(mapTracks, mapTracksPerView);

    ALICEVISION_LOG_INFO("Compute co-visibility");
    std::map<Pair, unsigned int> covisibility;
    computeCovisibility(covisibility, mapTracks);
    

    
    ALICEVISION_LOG_INFO("Process co-visibility");
    std::stringstream ss;
    ss << outputDirectory << "/pairs_" << rangeStart << ".json";
    std::ofstream of(ss.str());

    std::vector<sfm::ReconstructedPair> reconstructedPairs;

    double ratioChunk = double(covisibility.size()) / double(sfmData.getViews().size());
    int chunkStart = int(double(rangeStart) * ratioChunk);
    int chunkEnd = int(double(rangeStart + rangeSize) * ratioChunk);

    //For each covisible pair
#pragma omp parallel for
    for(int posPairs = chunkStart; posPairs < chunkEnd; posPairs++)
    {
        auto iterPairs = covisibility.begin();
        std::advance(iterPairs, posPairs);

        //Retrieve pair information
        IndexT refImage = iterPairs->first.first;
        IndexT nextImage = iterPairs->first.second;

        const sfmData::View& refView = sfmData.getView(refImage);
        const sfmData::View& nextView = sfmData.getView(nextImage);

        std::shared_ptr<camera::IntrinsicBase> refIntrinsics = sfmData.getIntrinsicsharedPtr(refView.getIntrinsicId());
        std::shared_ptr<camera::IntrinsicBase> nextIntrinsics = sfmData.getIntrinsicsharedPtr(nextView.getIntrinsicId());

        aliceVision::track::TracksMap mapTracksCommon;
        track::getCommonTracksInImagesFast({refImage, nextImage}, mapTracks, mapTracksPerView, mapTracksCommon);
        if (mapTracksCommon.size() == 0)
        {
            continue;
        }

        feature::MapFeaturesPerDesc& refFeaturesPerDesc = featuresPerView.getFeaturesPerDesc(refImage);
        feature::MapFeaturesPerDesc& nextFeaturesPerDesc = featuresPerView.getFeaturesPerDesc(nextImage);

        //Build features coordinates matrices
        const std::size_t n = mapTracksCommon.size();
        Mat refX(3, n);
        Mat nextX(3, n);
        std::vector<std::size_t> tracks;

        IndexT pos = 0;
        for(const auto& commonItem : mapTracksCommon)
        {
            const track::Track& track = commonItem.second;

            const feature::PointFeatures& refFeatures = refFeaturesPerDesc.at(track.descType);
            const feature::PointFeatures& nextfeatures = nextFeaturesPerDesc.at(track.descType);

            IndexT refFeatureId = track.featPerView.at(refImage);
            IndexT nextfeatureId = track.featPerView.at(nextImage);

            Vec2 refV = refFeatures[refFeatureId].coords().cast<double>();
            Vec2 nextV = nextfeatures[nextfeatureId].coords().cast<double>();

            //Lift to unit sphere
            refX.col(pos) = refIntrinsics->toUnitSphere(refIntrinsics->removeDistortion(refIntrinsics->ima2cam(refV)));
            nextX.col(pos) = nextIntrinsics->toUnitSphere(nextIntrinsics->removeDistortion(nextIntrinsics->ima2cam(nextV)));
            tracks.push_back(commonItem.first);

            pos++;
        }


        //Try to fit an essential matrix (we assume we are approx. calibrated)
        Mat3 R;
        std::vector<size_t> vecInliers;
        const bool relativeSuccess = robustRotation(R, vecInliers, refX, nextX, randomNumberGenerator, 1024, minInliers);
        if(!relativeSuccess)
        {
            continue;
        }

        sfm::ReconstructedPair reconstructed;
        reconstructed.reference = refImage;
        reconstructed.next = nextImage;
        reconstructed.R = R;
        reconstructed.t.fill(0);

        typedef bg::model::point<double, 2, bg::cs::cartesian> point_t;
        typedef bg::model::multi_point<point_t> mpoint_t;
        typedef boost::geometry::model::polygon<point_t> polygon;
        mpoint_t mpt1, mpt2;

        for (const auto & i : vecInliers)
        {
            std::size_t trackIndex = tracks[i];
            
            const track::Track& track = mapTracks[trackIndex];

            const feature::PointFeatures& refFeatures = refFeaturesPerDesc.at(track.descType);
            const feature::PointFeatures& nextfeatures = nextFeaturesPerDesc.at(track.descType);

            IndexT refFeatureId = track.featPerView.at(refImage);
            IndexT nextfeatureId = track.featPerView.at(nextImage);

            Vec2 refV = refFeatures[refFeatureId].coords().cast<double>();
            Vec2 nextV = nextfeatures[nextfeatureId].coords().cast<double>();

            bg::append(mpt1, point_t(refV(0), refV(1)));
            bg::append(mpt2, point_t(nextV(0), nextV(1)));
        }

        //Union
        double tracksUnion = mapTracksCommon.size();
        double tracksIntersection = vecInliers.size();
        double score = tracksIntersection / tracksUnion;

        polygon hull1, hull2;
        boost::geometry::convex_hull(mpt1, hull1);
        boost::geometry::convex_hull(mpt2, hull2);
        double area1 = bg::area(hull1);
        double area2 = bg::area(hull1);
        double score2 = (area1 + area2) / (refIntrinsics->w() * refIntrinsics->h() + nextIntrinsics->w() * nextIntrinsics->h());
        reconstructed.score = 0.5 * score + 0.5 * score2;


        //Buffered output to avoid lo
#pragma omp critical
        {

            reconstructedPairs.push_back(reconstructed);

            if(reconstructedPairs.size() > 1000)
            {
                boost::json::value jv = boost::json::value_from(reconstructedPairs);
                of << boost::json::serialize(jv);
                reconstructedPairs.clear();
            }
        }
    }

    //Serialize last pairs
    {
        boost::json::value jv = boost::json::value_from(reconstructedPairs);
        of << boost::json::serialize(jv);
    }

    of.close();

    return EXIT_SUCCESS;
}
