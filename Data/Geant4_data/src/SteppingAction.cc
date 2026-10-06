#include "SteppingAction.hh"
#include "RunAction.hh"
#include "G4Step.hh"
#include "G4VProcess.hh"
#include "G4VPhysicalVolume.hh"
#include "G4SystemOfUnits.hh"

SteppingAction::SteppingAction(RunAction* runAction)
: G4UserSteppingAction(), fRunAction(runAction)
{}

SteppingAction::~SteppingAction()
{}

void SteppingAction::UserSteppingAction(const G4Step* aStep)
{
    G4Track* track = aStep->GetTrack();

    // 1. We only care about interactions of the PRIMARY particle (TrackID = 1)
    //    If you want to track secondaries of secondaries, remove this check.
    if (track->GetTrackID() != 1) return;

    // 2. Check if secondaries were produced in this step
    const std::vector<const G4Track*>* secondaries = aStep->GetSecondaryInCurrentStep();
    if (secondaries->empty()) return;

    // 3. Geometry: keep only interactions inside the target, whatever its size.
    //    (Replaces the old |z| < 100 m cut, which matched only a 200 m target.)
    //    A step belongs to the volume of its pre-step point.
    const G4VPhysicalVolume* volume = aStep->GetPreStepPoint()->GetPhysicalVolume();
    if (!volume || volume->GetName() != "Target") return;

    // 4. Get Primary Energy BEFORE and AFTER the interaction
    G4double primaryPreEnergy = aStep->GetPreStepPoint()->GetKineticEnergy();
    G4double primaryPostEnergy = aStep->GetPostStepPoint()->GetKineticEnergy();

    // 5. Loop over generated secondaries to find Nuclei/Neutrons
    for (auto& sec : *secondaries) {

        const G4ParticleDefinition* particle = sec->GetDefinition();
        G4String name = particle->GetParticleName();
        G4double secEnergy = sec->GetKineticEnergy();
        G4double zPos = sec->GetPosition().z();

        // Filter: Ignore electrons
        if (name == "e-") continue;

        // Filter: Threshold Energy
        G4double minEnergy = 1000.0 * eV;
        if (secEnergy <= minEnergy) continue;

        // Filter: Select heavy particles (nuclei, nucleons, ...)
        if (particle->GetPDGMass()/MeV > 900.) {

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
