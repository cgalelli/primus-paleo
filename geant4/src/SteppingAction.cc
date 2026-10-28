#include "SteppingAction.hh"
#include "RunAction.hh"
#include "G4Step.hh"
#include "G4VProcess.hh"
#include "G4VPhysicalVolume.hh"
#include "G4SystemOfUnits.hh"
 
#include <cstdlib>
#include <fstream>
#include <iomanip>
 
SteppingAction::SteppingAction(RunAction* runAction)
: G4UserSteppingAction(), fRunAction(runAction)
{}
 
SteppingAction::~SteppingAction()
{}
 
void SteppingAction::UserSteppingAction(const G4Step* aStep)
{
    G4Track* track = aStep->GetTrack();
 
    if (track->GetTrackID() != 1) return;
 
    const std::vector<const G4Track*>* secondaries = aStep->GetSecondaryInCurrentStep();
    if (secondaries->empty()) return;
 
    const G4VPhysicalVolume* volume = aStep->GetPreStepPoint()->GetPhysicalVolume();
    if (!volume || volume->GetName() != "Target") return;
 
    G4double primaryPreEnergy = aStep->GetPreStepPoint()->GetKineticEnergy();
    G4double primaryPostEnergy = aStep->GetPostStepPoint()->GetKineticEnergy();
 
    const G4int primaryPDG = std::abs(track->GetDefinition()->GetPDGEncoding());
    const G4bool primaryIsNeutrino = (primaryPDG == 12 || primaryPDG == 14 || primaryPDG == 16);
 
    for (auto& sec : *secondaries) {
 
        const G4ParticleDefinition* particle = sec->GetDefinition();
        G4String name = particle->GetParticleName();
        G4double secEnergy = sec->GetKineticEnergy();
        G4double zPos = sec->GetPosition().z();
 
        if (name == "e-") continue;
 
        G4double minEnergy = 1000.0 * eV;
        if (secEnergy <= minEnergy) continue;
 
        const G4bool isHeavy = particle->GetPDGMass()/MeV > 900.;
        const G4bool isMuon  = std::abs(particle->GetPDGEncoding()) == 13;
        if (isHeavy || (primaryIsNeutrino && isMuon)) {
 
            std::ofstream* out = fRunAction->GetOutputFile();
            if (out && out->is_open()) {
                *out << std::setprecision(10);
                *out << name << "\t"
                     << particle->GetPDGMass()/MeV << "\t"
                     << secEnergy/MeV << "\t"
                     << zPos/mm << "\t"
                     << primaryPreEnergy/MeV << "\t"
                     << primaryPostEnergy/MeV << G4endl;
            }
        }
    }
}