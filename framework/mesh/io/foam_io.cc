// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mesh/io/mesh_io.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"
#include <fstream>
#include "argList.H"
#include "word.H"
#include "fvMesh.H"
#include "polyMesh.H"
#include "IOobject.H"
#include "Time.H"


namespace opensn
{


int main(int argc, char** argv) {
    // Initialize OpenFOAM argument handling
    Foam::argList::validArgs.append("case");
    Foam::argList::noMandatoryArgs();
    Foam::argList args(argc, argv);

    // Get the case directory
    Foam::word caseDir = args.getOrDefault<Foam::word>("case", ".");
    Foam::Info << "Case directory: " << caseDir << Foam::nl;
    Foam::fileName fullCaseDir = Foam::fileName(caseDir).expand();
    Foam::Info << "Full case directory: " << fullCaseDir << Foam::nl;

    // Initialize runTime
    Foam::Time runTime(Foam::Time::controlDictName, args);
    Foam::Info << "Run time initialized: " << runTime.timeName() << Foam::nl;

    // Verify the presence of the "points" file in constant/polyMesh
    Foam::fileName pointsPath = runTime.constant() / Foam::polyMesh::meshSubDir / "points";
    Foam::Info << "Checking for file: " << pointsPath << Foam::nl;
    if (!Foam::isFile(pointsPath)) {
        Foam::Info << "Error: Mesh not found " << Foam::nl;
        return 1; // Exit with an error code
    }

    Foam::Info << "Mesh file 'points' exists at: "
               << pointsPath << Foam::nl;

    // Construct mesh (i.e. read mesh)
    try {
        Foam::fvMesh mesh(
            Foam::IOobject(
                Foam::fvMesh::defaultRegion,
                runTime.timeName(),
                runTime,
                Foam::IOobject::MUST_READ
            )
        );

        // Output the number of cells and points in the mesh
        Foam::Info << "Number of cells: " << mesh.nCells() << Foam::nl;
        Foam::Info << "Number of points: " << mesh.nPoints() << Foam::nl;

    } catch (const Foam::error& e) {
        Foam::Info << "Error while reading the mesh: " << e << Foam::nl;
        return 1;
    }

    Foam::Info << "Mesh successfully read from: " << caseDir << Foam::nl;
    return 0;
}



} // namespace opensn