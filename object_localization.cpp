#include <iostream>
#include <thread>
#include <chrono>

#include "carla/client/Client.h"
#include "carla/client/World.h"
#include "carla/client/ActorBlueprint.h"
#include "carla/client/BlueprintLibrary.h"
#include "carla/client/Sensor.h"
#include "carla/client/Vehicle.h"

#include "carla/sensor/data/Image.h"

int main()
{
    // Connect to CARLA
    

    carla::client::Client client("localhost", 2000);

    client.SetTimeout(std::chrono::seconds(10));

    std::cout << "Connecting to CARLA..." << std::endl;

    auto world = client.GetWorld();

    std::cout << "Connected!" << std::endl;

    // Get blueprint library
    auto blueprint_library = world.GetBlueprintLibrary();

    //Get a vehicle blueprint
    auto vehicle_blueprint = (*blueprint_library)->Find("vehicle.lincoln.mkz_2020");

    //Get a spawn point

    auto spawn_points = world.GetMap()->GetRecommendedSpawnPoints();

    if (spawn_points.empty())
    {
        std::cerr << "No spawn points available." << std::endl;
        return 1;
    }

    auto transform = spawn_points[0];

    // Spawn vehicle

    auto vehicle = boost::static_pointer_cast<carla::client::Vehicle>(world.SpawnActor(*vehicle_blueprint, transform));

    if (!vehicle)
    {
        std::cerr << "Failed to spawn vehicle." << std::endl;
        return 1;
    }

    std::cout << "Vehicle spawned!" << std::endl;

    //Get RGB camera blueprint

    auto camera_blueprint = (*blueprint_library)->Find("sensor.camera.rgb");

    // Configure camera

    camera_blueprint.SetAttribute("image_size_x", "800");
    camera_blueprint.SetAttribute("image_size_y", "600");
    camera_blueprint.SetAttribute("fov", "90");

    // Position camera relative to vehicle

    carla::geom::Transform camera_transform(
        carla::geom::Location(
            1.5f,   // x
            0.0f,   // y
            1.6f    // z
        )
    );

    // Spawn camera attached to vehicle

    auto camera = boost::static_pointer_cast<carla::client::Sensor>(
        world.SpawnActor(
            *camera_blueprint,
            camera_transform,
            vehicle.get()
        )
    );

    if (!camera)
    {
        std::cerr << "Failed to spawn camera." << std::endl;

        vehicle->Destroy();

        return 1;
    }

    std::cout << "Camera spawned!" << std::endl;


    // --------------------------------------------------
    // 10. Receive images
    // --------------------------------------------------

    camera->Listen(
        [](auto image)
        {
            std::cout
                << "Frame: "
                << image->GetFrame()
                << " | Timestamp: "
                << image->GetTimestamp()
                << " | Resolution: "
                << image->GetWidth()
                << " x "
                << image->GetHeight()
                << std::endl;


            // Save image every frame
            std::string filename =
                "image_" +
                std::to_string(image->GetFrame()) +
                ".png";

            image->SaveToDisk(filename);

            std::cout
                << "Saved: "
                << filename
                << std::endl;
        }
    );


    // --------------------------------------------------
    // 11. Keep program alive
    // --------------------------------------------------

    std::cout << "Camera running..." << std::endl;

    while (true)
    {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(100)
        );
    }


    // --------------------------------------------------
    // 12. Cleanup
    // --------------------------------------------------

    camera->Stop();
    camera->Destroy();
    vehicle->Destroy();

    return 0;
}