#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "StackingAction.hh"
#include "SteppingAction.hh"

// --- Headers required for dE/dx extraction ---
#include "G4EmCalculator.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4IonTable.hh"
#include "G4SystemOfUnits.hh"
#include <fstream>
#include <iomanip>
#include <string>

// --- Function to export the stopping power tables ---
void ExportStoppingPowerTable(const G4String& mineralName, int Z, int A, const G4String& outputDir) {
    // 1. Try to get the material from custom definitions first, fallback to NIST
    G4Material* material = G4Material::GetMaterial(mineralName);
    if (!material) {
        material = G4NistManager::Instance()->FindOrBuildMaterial(mineralName);
    }
    
    if (!material) {
        G4cout << "Error: Material " << mineralName << " not found!" << G4endl;
        return;
    }

    G4ParticleDefinition* ion = G4IonTable::GetIonTable()->GetIon(Z, A, 0.0);
    if (!ion) {
        G4cout << "Error: Ion Z=" << Z << " A=" << A << " could not be created." << G4endl;
        return;
    }

    G4EmCalculator emCalc;
    
    std::string filename = outputDir + "/DEDX_Z" + std::to_string(Z) + "_A" + std::to_string(A) + ".txt";
    std::ofstream outFile(filename);

    outFile << "# Energy_MeV\tdEdx_Elec_MeV_um\tdEdx_Nucl_MeV_um\tRange_um\n";
    outFile << std::scientific << std::setprecision(6);

  for (double eKin = 0.00001 * MeV; eKin <= 10000.0 * MeV; eKin *= 1.15) {
          double dedxElec = emCalc.ComputeElectronicDEDX(eKin, ion, material) / (MeV / um);
          double dedxNucl = emCalc.ComputeNuclearDEDX(eKin, ion, material) / (MeV / um);
          
          double range = emCalc.GetCSDARange(eKin, ion, material) / um;

          outFile << eKin / MeV << "\t" 
                  << dedxElec << "\t" 
                  << dedxNucl << "\t" 
                  << range << "\n";
      }
    outFile.close();
}


int main(int argc, char** argv) {
    if (argc < 2) {
        G4cout << "Usage for simulation: " << argv[0] << " <macro_file>" << G4endl;
        G4cout << "Usage for dedx export: " << argv[0] << " --export-dedx <mineral> <Z> <A> <outDir> [materials.mac]" << G4endl;
        return 1;
    }

    G4RunManager* runManager = new G4RunManager;

    runManager->SetUserInitialization(new DetectorConstruction());
    runManager->SetUserInitialization(new PhysicsList());

    runManager->SetUserAction(new PrimaryGeneratorAction());
    
    RunAction* runAction = new RunAction();
    runManager->SetUserAction(runAction);
    
    runManager->SetUserAction(new EventAction());
    runManager->SetUserAction(new StackingAction());
    runManager->SetUserAction(new SteppingAction(runAction));

    G4String arg1 = argv[1];
    if (arg1 == "--export-dedx" && argc >= 6) {
        G4String mineral = argv[2];
        int Z = std::stoi(argv[3]);
        int A = std::stoi(argv[4]);
        G4String outDir = argv[5];
        
        G4UImanager* UI = G4UImanager::GetUIpointer();
        
        UI->ApplyCommand("/control/verbose 0");
        UI->ApplyCommand("/run/verbose 0");
        
        // Optional 6th argument: macro with /testhadr/defineMaterial commands
        if (argc >= 7) {
            G4int status = UI->ApplyCommand(G4String("/control/execute ") + argv[6]);
            if (status != 0) {
                G4cout << "Error: could not execute materials macro " << argv[6]
                       << " (status " << status << ")" << G4endl;
                delete runManager;
                return 1;
            }
        }

        UI->ApplyCommand("/testhadr/TargetMat " + mineral);
        UI->ApplyCommand("/run/setCut 0.005 mm");
        UI->ApplyCommand("/process/eLoss/CSDARange true");
        UI->ApplyCommand("/run/initialize");
        UI->ApplyCommand("/gun/particle ion");
        UI->ApplyCommand("/gun/ion " + std::to_string(Z) + " " + std::to_string(A) + " " + std::to_string(Z) + " 0.0");
        UI->ApplyCommand("/run/beamOn 0"); 
        
        ExportStoppingPowerTable(mineral, Z, A, outDir);
        
        delete runManager;
        return 0;
    }

    G4UImanager* UImanager = G4UImanager::GetUIpointer();
    G4String command = "/control/execute ";
    G4String fileName = argv[1];
    UImanager->ApplyCommand(command + fileName);

    delete runManager;
    return 0;
}