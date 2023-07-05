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

#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/feature/imageDescriberCommon.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <aliceVision/robustEstimation/ACRansac.hpp>
#include <aliceVision/multiview/RelativePoseKernel.hpp>
#include <aliceVision/multiview/relativePose/Rotation3PSolver.hpp>

#include <aliceVision/sfm/pipeline/relativePoses.hpp>
#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

#include <aliceVision/track/tracksUtils.hpp>
#include <aliceVision/track/trackIO.hpp>

#include <aliceVision/stl/mapUtils.hpp>
#include <aliceVision/sfm/liealgebra.hpp>


#include <cstdlib>
#include <random>
#include <regex>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
/*#include <boost/graph/kruskal_min_spanning_tree.hpp>*/

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

using namespace aliceVision;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

std::vector<boost::json::value> readJsons(std::istream& is, boost::json::error_code& ec)
{
    std::vector<boost::json::value> jvs;
    boost::json::stream_parser p;
    std::string line;
    std::size_t n = 0;


    while(true)
    {
        if(n == line.size())
        {
            if(!std::getline(is, line))
            {
                break;
            }

            n = 0;
        }

        //Consume at least part of the line
        n += p.write_some( line.data() + n, line.size() - n, ec);

        //If the parser found a value, add it
        if (p.done())
        {
            jvs.push_back(p.release());
            p.reset();
        }
    }

    if (!p.done())
    {
        //Try to extract the end
        p.finish(ec);
        if (ec.failed())
        {
            return jvs;
        }

        jvs.push_back(p.release());
    }

    return jvs;
}


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

int aliceVision_main(int argc, char** argv)
{
    // command-line parameters
    std::string sfmDataFilename;
    std::string sfmDataOutputFilename;
    std::vector<std::string> featuresFolders;
    std::string tracksFilename;
    std::string pairsDirectory;

    // user optional parameters
    std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);
    std::pair<std::string, std::string> initialPairString("", "");

    const double maxEpipolarDistance = 4.0;
    const double minAngle = 5.0;

    int randomSeed = std::mt19937::default_seed;

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
    ("input,i", po::value<std::string>(&sfmDataFilename)->required(), "SfMData file.")
    ("output,o", po::value<std::string>(&sfmDataOutputFilename)->required(), "SfMData output file.")
    ("tracksFilename,t", po::value<std::string>(&tracksFilename)->required(), "Tracks file.")
    ("pairs,p", po::value<std::string>(&pairsDirectory)->required(), "Path to the pairs directory.")
    ("featuresFolders,f", po::value<std::vector<std::string>>(&featuresFolders)->multitoken(), "Path to folder(s) containing the extracted features.")
    ("describerTypes,d", po::value<std::string>(&describerTypesName)->default_value(describerTypesName),feature::EImageDescriberType_informations().c_str());

    CmdLine cmdline("AliceVision pairsEstimations");

    cmdline.add(requiredParams);
    if(!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    // set maxThreads
    HardwareContext hwc = cmdline.getHardwareContext();
    omp_set_num_threads(hwc.getMaxThreads());
    
    // load input SfMData scene
    sfmData::SfMData sfmData;
    if(!sfmDataIO::Load(sfmData, sfmDataFilename, sfmDataIO::ESfMData::ALL))
    {
        ALICEVISION_LOG_ERROR("The input SfMData file '" + sfmDataFilename + "' cannot be read.");
        return EXIT_FAILURE;
    }


    if (sfmData.getValidViews().size() >= 2)
    {
        ALICEVISION_LOG_INFO("SfmData has already an initialization");
        return EXIT_SUCCESS;
    }


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


    //Result of pair estimations are stored in multiple files
    std::vector<sfm::ReconstructedPair> reconstructedPairs;
    const std::regex regex("pairs\\_[0-9]+\\.json");
    for(fs::directory_entry & file : boost::make_iterator_range(fs::directory_iterator(pairsDirectory), {}))
    {
        if (!std::regex_search(file.path().string(), regex))
        {
            continue;
        }

        std::ifstream inputfile(file.path().string());        

        boost::json::error_code ec;
        std::vector<boost::json::value> values = readJsons(inputfile, ec);
        for (const boost::json::value & value : values)
        {
            std::vector<sfm::ReconstructedPair> localVector = boost::json::value_to<std::vector<sfm::ReconstructedPair>>(value);
            reconstructedPairs.insert(reconstructedPairs.end(), localVector.begin(), localVector.end());
        }
    }

    if (reconstructedPairs.size() == 0)
    {
        ALICEVISION_LOG_ERROR("No precomputed pairs found");
        return EXIT_FAILURE;
    }

    //Sort reconstructedPairs by quality
    std::sort(reconstructedPairs.begin(), reconstructedPairs.end(), 
        [](const sfm::ReconstructedPair& p1, const sfm::ReconstructedPair & p2)
        {
            return p1.score > p2.score;
        }
    );

    /*struct VertexProperties
    {
        IndexT viewId;
    };


    typedef boost::property<boost::edge_weight_t, double> EdgeWeightProperty;
    typedef boost::property<boost::vertex_name_t, IndexT> vertex_property_t;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperties, EdgeWeightProperty> graph_t;
    typedef graph_t::vertex_descriptor vertex_t;
    typedef graph_t::edge_descriptor edge_t;
    typedef boost::graph_traits<graph_t>::edge_iterator edge_iter;

    graph_t g; 

    boost::property_map< graph_t, boost::edge_weight_t >::type weight = boost::get(boost::edge_weight, g);

    std::map<IndexT, vertex_t> nodes;
    std::map<vertex_t, IndexT> nodes_inverse;
    for (const auto& pv : sfmData.getViews())
    {
        vertex_t id = boost::add_vertex({ pv.first }, g);

        nodes[pv.first] = id;
        nodes_inverse[id] = pv.first;
    }

    for (const auto & pr : reconstructedPairs)
    {
        const vertex_t & v1 = nodes[pr.reference];
        const vertex_t & v2 = nodes[pr.next];
        boost::add_edge(v1, v2, {1.0 - pr.score}, g);
    }

    std::vector<vertex_t> p(num_vertices(g));
    std::vector<double> d(num_vertices(g));


    std::map<Pair, double> distances;

    for (const auto& pv : sfmData.getViews())
    {
        vertex_t s = nodes[pv.first];

        boost::dijkstra_shortest_paths(
                                        g, s,
                                        predecessor_map(
                                            boost::make_iterator_property_map(
                                                p.begin(), get(boost::vertex_index, g)
                                            )
                                        )
                                        .distance_map(
                                            boost::make_iterator_property_map(
                                                d.begin(), get(boost::vertex_index, g)
                                            )
                                        )
                                    );

        for (int i = 0 ; i < p.size(); i++)
        {
            IndexT view = nodes_inverse[i];
            Pair pair = std::make_pair(pv.first, view);

            distances[pair] = d[i];
        }
    }

    
    double t = 50.0;
    double rho = 4.0 / t;

    
    std::vector<sfm::ReconstructedPair> filteredPairs;
    for (auto & p : reconstructedPairs)
    {
        Pair pair = std::make_pair(p.reference, p.next);

        //Importance is the ratio shortest_distance to edge.
        //The smaller the value, the less this edge is significant
        double imp = distances[pair] / (1.0 - p.score);
        if (imp < rho)
        {
            continue;
        }

        //Remove small rotations
        double angle = SO3::logm(p.R).norm();
        if (angle < 1.0 * M_PI/180.0) continue;

        filteredPairs.push_back(p);
    }


    std::cout << filteredPairs.size() << std::endl;
   
    std::map<IndexT, std::vector<IndexT>> connections;
    for (auto & p : filteredPairs)
    {
        connections[p.reference].push_back(p.next);
        connections[p.next].push_back(p.reference);
    }

    enum VColor
    {
        white,
        grey,
        black
    };
    
    std::map<IndexT, VColor> colors;
    for (auto & p: connections)
    {
        colors[p.first] = VColor::white;
    }

    while (1)
    {
        //Select the next vertex
        int max = 0;
        IndexT selected = UndefinedIndexT;

        for (auto & p : connections)
        {
            int count = 0;
            for (IndexT ido : p.second)
            {
                if (colors[ido] == VColor::white)
                {
                    count++;
                }
            }

            if (count > max)
            {
                max = count;
                selected = p.first;
            }
        }

        if (selected == UndefinedIndexT)
        {
            break;
        }

        colors[selected] = VColor::black;
        for (IndexT ido : connections[selected])
        {
            if (colors[ido] == VColor::white)
            {
                colors[ido] = VColor::grey;
            }
        }
    }

    int count = 0;
    for (auto c : colors)
    {
        if (c.second == VColor::black)
        {
            count++;
        }
    }
    std::cout << count << std::endl;*/

    std::map<IndexT, Eigen::Matrix3d> reconstructedViews;
    std::map<size_t, Vec3> worldPoints;

    IndexT refViewId = reconstructedPairs[0].reference;
    reconstructedViews[refViewId] = Eigen::Matrix3d::Identity();

    const Eigen::Matrix3d & R = reconstructedViews[refViewId];
    const sfmData::View& refView = sfmData.getView(refViewId);
    std::shared_ptr<camera::IntrinsicBase> refIntrinsics = sfmData.getIntrinsicsharedPtr(refView.getIntrinsicId());
    feature::MapFeaturesPerDesc& refFeaturesPerDesc = featuresPerView.getFeaturesPerDesc(refViewId);

    for (const auto id : mapTracksPerView[refViewId])
    {
        const track::Track& track = mapTracks[id];
        if (worldPoints.find(id) != worldPoints.end())
        {
            continue;
        }

        const feature::PointFeatures& refFeatures = refFeaturesPerDesc.at(track.descType);
        IndexT refFeatureId = track.featPerView.at(refViewId);
        Vec2 refV = refFeatures[refFeatureId].coords().cast<double>();
        Vec3 camP = refIntrinsics->toUnitSphere(refIntrinsics->ima2cam(refIntrinsics->get_ud_pixel(refV)));
        Vec3 worldP = R.transpose() * camP;
        worldPoints[id] = worldP;
    }

    bool doSth = true;
    while (doSth)
    {
        doSth = false;
        for (const auto & pair : reconstructedPairs)
        {
            bool refFound = (reconstructedViews.find(pair.reference) != reconstructedViews.end());
            bool nextFound = (reconstructedViews.find(pair.next) != reconstructedViews.end());

            if (refFound && nextFound) continue;
            if (!(refFound || nextFound)) continue;

            IndexT newViewId = refFound ? pair.next : pair.reference;

            std::set<size_t> existingTracks;
            std::transform(worldPoints.begin(), worldPoints.end(), std::inserter(existingTracks, existingTracks.begin()), stl::RetrieveKey());

            const aliceVision::track::TrackIdSet& newViewTracks = mapTracksPerView.at(newViewId);
        
            std::vector<IndexT> observedTracks;
            std::set_intersection(existingTracks.begin(), existingTracks.end(), newViewTracks.begin(), newViewTracks.end(), std::back_inserter(observedTracks));

            const sfmData::View& newView = sfmData.getView(newViewId);
            std::shared_ptr<camera::IntrinsicBase> newViewIntrinsics = sfmData.getIntrinsicsharedPtr(newView.getIntrinsicId());
            feature::MapFeaturesPerDesc& newViewFeaturesPerDesc = featuresPerView.getFeaturesPerDesc(newViewId);
            
            Mat refX(3, observedTracks.size());
            Mat newX(3, observedTracks.size());

            int pos = 0;
            for (IndexT trackId : observedTracks)
            {
                const track::Track& track = mapTracks[trackId];

                const feature::PointFeatures& newViewFeatures = newViewFeaturesPerDesc.at(track.descType);
                IndexT newViewFeatureId = track.featPerView.at(newViewId);
                Vec2 nvV = newViewFeatures[newViewFeatureId].coords().cast<double>();
                Vec3 camP = newViewIntrinsics->toUnitSphere(newViewIntrinsics->ima2cam(newViewIntrinsics->get_ud_pixel(nvV)));

                refX.col(pos) = worldPoints[trackId];
                newX.col(pos) = camP;

                pos++;
            }

            Mat3 R;
            std::vector<size_t> vecInliers;
            const size_t minInliers = 35;
            std::mt19937 randomNumberGenerator(randomSeed);
            const bool relativeSuccess = robustRotation(R, vecInliers, refX, newX, randomNumberGenerator, 1024, minInliers);
            if(!relativeSuccess)
            {
                continue;
            }

            reconstructedViews[newViewId] = R;

            for (const auto id : mapTracksPerView[newViewId])
            {
                const track::Track& track = mapTracks[id];
                if (worldPoints.find(id) != worldPoints.end())
                {
                    continue;
                }

                const feature::PointFeatures& newViewFeatures = newViewFeaturesPerDesc.at(track.descType);
                IndexT newViewFeatureId = track.featPerView.at(newViewId);
                Vec2 nvV = newViewFeatures[newViewFeatureId].coords().cast<double>();
                Vec3 camP = newViewIntrinsics->toUnitSphere(newViewIntrinsics->ima2cam(newViewIntrinsics->get_ud_pixel(nvV)));
                Vec3 worldP = R.transpose() * camP;

                worldPoints[id] = worldP;
            }
            
            std::cout << worldPoints.size() << std::endl;
            doSth = true;
            break;
        }
    }


    /*std::map<IndexT, Eigen::Matrix3d> reconstructedViews;

    //Bootstrap
    IndexT ref = reconstructedPairs[0].reference;
    reconstructedViews[ref] = Eigen::Matrix3d::Identity();
    
    size_t count = reconstructedViews.size();
    size_t previousCount = 0;

    while (previousCount < count)
    {
        previousCount = count;
        for (const auto & pair : reconstructedPairs)
        {
            bool refFound = (reconstructedViews.find(pair.reference) != reconstructedViews.end());
            bool nextFound = (reconstructedViews.find(pair.next) != reconstructedViews.end());

            if (refFound && nextFound) continue;
            if (!(refFound || nextFound)) continue;

            IndexT newId = refFound ? pair.next : pair.reference;


            const sfmData::View& nextView = sfmData.getView(newId);
            std::shared_ptr<camera::IntrinsicBase> nextIntrinsics = sfmData.getIntrinsicsharedPtr(nextView.getIntrinsicId());

            std::vector<Vec3> referencePoints;
            std::vector<Vec3> currentPoints;

            for (const auto & otherView : reconstructedViews)
            {
                aliceVision::track::TracksMap mapTracksCommon;
                track::getCommonTracksInImagesFast({newId, otherView.first}, mapTracks, mapTracksPerView, mapTracksCommon);

                if (mapTracksCommon.size() == 0) continue;

                const Eigen::Matrix3d & R = otherView.second;

                const sfmData::View& refView = sfmData.getView(otherView.first);
                std::shared_ptr<camera::IntrinsicBase> refIntrinsics = sfmData.getIntrinsicsharedPtr(refView.getIntrinsicId());

                feature::MapFeaturesPerDesc& refFeaturesPerDesc = featuresPerView.getFeaturesPerDesc(newId);
                feature::MapFeaturesPerDesc& nextFeaturesPerDesc = featuresPerView.getFeaturesPerDesc(otherView.first);

                for(const auto& commonItem : mapTracksCommon)
                {
                    const track::Track& track = commonItem.second;

                    const feature::PointFeatures& refFeatures = refFeaturesPerDesc.at(track.descType);
                    const feature::PointFeatures& nextfeatures = nextFeaturesPerDesc.at(track.descType);

                    IndexT refFeatureId = track.featPerView.at(newId);
                    IndexT nextfeatureId = track.featPerView.at(otherView.first);

                    Vec2 refV = refFeatures[refFeatureId].coords().cast<double>();
                    Vec2 nextV = nextfeatures[nextfeatureId].coords().cast<double>();

                    //Lift to unit sphere
                    Vec3 refpt = refIntrinsics->toUnitSphere(refIntrinsics->removeDistortion(refIntrinsics->ima2cam(refV)));
                    Vec3 curpt = nextIntrinsics->toUnitSphere(nextIntrinsics->removeDistortion(nextIntrinsics->ima2cam(nextV)));

                    referencePoints.push_back(R.transpose() * refpt);
                    currentPoints.push_back(curpt);
                }
            }

            Mat refX(3, referencePoints.size());
            Mat nextX(3, currentPoints.size());

            for (int i = 0; i <  referencePoints.size(); i++)
            {
                refX.col(i) = referencePoints[i];
                nextX.col(i) = currentPoints[i];
            }

            //Try to fit an essential matrix (we assume we are approx. calibrated)
            Mat3 R;
            std::vector<size_t> vecInliers;
            const size_t minInliers = 35;
            std::mt19937 randomNumberGenerator(randomSeed);
            const bool relativeSuccess = robustRotation(R, vecInliers, refX, nextX, randomNumberGenerator, 1024, minInliers);
            if(!relativeSuccess)
            {
                continue;
            }

            reconstructedViews[newId] = R;
            std::cout << "ok" << std::endl;
            break;
        }

        count = reconstructedViews.size();
        std::cout << count << std::endl;
    }*/

    sfmDataIO::Save(sfmData, sfmDataOutputFilename, sfmDataIO::ESfMData::ALL);

    return EXIT_SUCCESS;
}


/*struct VertexProperties
    {
        IndexT viewId;
    };


    typedef boost::property<boost::edge_weight_t, double> EdgeWeightProperty;
    typedef boost::property<boost::vertex_name_t, IndexT> vertex_property_t;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperties, EdgeWeightProperty> graph_t;
    typedef graph_t::vertex_descriptor vertex_t;
    typedef graph_t::edge_descriptor edge_t;
    typedef boost::graph_traits<graph_t>::edge_iterator edge_iter;

    graph_t g; 

    boost::property_map< graph_t, boost::edge_weight_t >::type weight = boost::get(boost::edge_weight, g);

    std::set<IndexT> roots;
    std::map<IndexT, vertex_t> nodes;
    std::map<vertex_t, IndexT> nodes_inverse;
    for (const auto& pv : sfmData.getViews())
    {
        vertex_t id = boost::add_vertex({ pv.first }, g);

        nodes[pv.first] = id;
        nodes_inverse[id] = pv.first;
    }

    for (const auto & pr : reconstructedPairs)
    {
        const vertex_t & v1 = nodes[pr.reference];
        const vertex_t & v2 = nodes[pr.next];

        boost::add_edge(v1, v2, {1.0 - pr.score}, g);

        roots.insert(v1);
        roots.insert(v2);
    }
    
    std::vector<edge_t> spanning_tree;
    boost::kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

    std::map<IndexT, std::vector<IndexT>> tree;
    
    std::set<IndexT> uniques;
    for (std::vector<edge_t>::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei)
    {
        vertex_t v1 = source(*ei, g); 
        vertex_t v2 = target(*ei, g);
        IndexT vid1 = nodes_inverse[v1];
        IndexT vid2 = nodes_inverse[v2];

        if (uniques.find(v2) != uniques.end())
        {
            std::cout << "what" << std::endl;
        }
        uniques.insert(v2);

        tree[vid1].push_back(vid2);
        roots.erase(vid2);
    }

    std::cout << spanning_tree.size() << " " << tree.size() << std::endl;*/