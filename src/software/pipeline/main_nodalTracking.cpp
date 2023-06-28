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

#include <aliceVision/sfm/pipeline/relativePoses.hpp>
#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

#include <aliceVision/track/tracksUtils.hpp>
#include <aliceVision/track/trackIO.hpp>


#include <cstdlib>
#include <random>
#include <regex>

/*#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>*/

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


    std::map<IndexT, Eigen::Matrix3d> reconstructedViews;

    //Bootstrap
    IndexT ref = reconstructedPairs[0].reference;
    reconstructedViews[ref] = Eigen::Matrix3d::Identity();
    
    for (const auto & pair : reconstructedPairs)
    {
        bool refFound = (reconstructedViews.find(pair.reference) != reconstructedViews.end());
        bool nextFound = (reconstructedViews.find(pair.next) != reconstructedViews.end());

        if (refFound && nextFound) continue;
        if (!(refFound || nextFound)) continue;

        IndexT newId = refFound ? pair.reference : pair.next;

        for (const auto & otherView : reconstructedViews)
        {
            aliceVision::track::TracksMap mapTracksCommon;
            track::getCommonTracksInImagesFast({newId, otherView.first}, mapTracks, mapTracksPerView, mapTracksCommon);
        }
    }

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