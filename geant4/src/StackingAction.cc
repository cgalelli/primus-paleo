#include "StackingAction.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"

StackingAction::StackingAction()
  : G4UserStackingAction()
{}

StackingAction::~StackingAction() {}

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track* aTrack) {
    
    const G4ParticleDefinition* particle = aTrack->GetDefinition();
    G4String name = particle->GetParticleName();
    
    // PERFORMANCE OPTIMIZATION:
    // Kill electrons immediately. We don't record them, and tracking them takes forever.
    if (name == "e-" || name == "e+") return fKill; 

    // Track everything else (Nuclei, Neutrons, Gammas, etc.)
    return fUrgent;
}