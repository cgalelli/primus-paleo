#include "PrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Tubs.hh"
#include "G4Exception.hh"

PrimaryGeneratorAction::PrimaryGeneratorAction()
 : G4VUserPrimaryGeneratorAction(),
   fParticleGun(0)
{
  fParticleGun  = new G4ParticleGun(1);
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.,0.,-1.));
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
  // The gun follows the current target geometry: it sits just above the
  // top face of the target (in the vacuum world), pointing down (-z).
  // The target is placed at the world origin, so its top face is at
  // z = +HalfLength. Looked up every event so /testhadr/TargetLength
  // changes between runs are picked up automatically.
  G4LogicalVolume* lv = G4LogicalVolumeStore::GetInstance()->GetVolume("Target", false);
  G4Tubs* target = lv ? dynamic_cast<G4Tubs*>(lv->GetSolid()) : nullptr;
  if (!target) {
    G4Exception("PrimaryGeneratorAction::GeneratePrimaries", "Gun001",
                FatalException, "Target volume (G4Tubs \"Target\") not found.");
    return;
  }

  const G4double zTop = target->GetZHalfLength();
  fParticleGun->SetParticlePosition(G4ThreeVector(0., 0., zTop + 1.*mm));
  fParticleGun->GeneratePrimaryVertex(anEvent);
}
