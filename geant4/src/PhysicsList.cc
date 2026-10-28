#include "PhysicsList.hh"
 
#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4ParticleTypes.hh"
#include "G4ParticleTable.hh"
#include "G4Version.hh"
 
// Physics Constructors
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmExtraPhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4HadronPhysicsQGSP_BIC_HP.hh"
#include "G4StoppingPhysics.hh"
#include "G4IonPhysics.hh"
#if G4VERSION_NUMBER >= 1120
#include "G4NeutrinoPhysics.hh"
#endif

static const G4String kNuRegionName = "TargetRegion";
static const G4double kNuBias = 1.e6;
 
PhysicsList::PhysicsList() : G4VModularPhysicsList()
{
  SetVerboseLevel(0);
 
  RegisterPhysics(new G4EmStandardPhysics_option4(0));
 
#if G4VERSION_NUMBER >= 1120
  RegisterPhysics(new G4EmExtraPhysics());
 
  auto* nuPhys = new G4NeutrinoPhysics();
  nuPhys->SetNuDetectorName(kNuRegionName);
  nuPhys->NuETotXscActivated(true);
  nuPhys->SetNuNucleusBias(kNuBias);
  nuPhys->SetNuEleCcBias(kNuBias);
  nuPhys->SetNuEleNcBias(kNuBias);
  RegisterPhysics(nuPhys);
#else
  auto* extraPhys = new G4EmExtraPhysics();
  extraPhys->NeutrinoActivated(true);
  extraPhys->SetNuDetectorName(kNuRegionName);
  extraPhys->NuETotXscActivated(true);
  extraPhys->SetNuNucleusBias(kNuBias);
  extraPhys->SetNuEleCcBias(kNuBias);
  extraPhys->SetNuEleNcBias(kNuBias);
  RegisterPhysics(extraPhys);
#endif
 
  RegisterPhysics(new G4DecayPhysics(0));
  RegisterPhysics(new G4RadioactiveDecayPhysics(0));
 
  RegisterPhysics(new G4HadronElasticPhysics(0));
 
  RegisterPhysics(new G4HadronPhysicsQGSP_BIC_HP(0));
 
  RegisterPhysics(new G4StoppingPhysics(0));
 
  RegisterPhysics(new G4IonPhysics(0));
}
 
PhysicsList::~PhysicsList()
{}
 
void PhysicsList::ConstructParticle()
{
  G4VModularPhysicsList::ConstructParticle();
}
 
void PhysicsList::ConstructProcess()
{
  G4VModularPhysicsList::ConstructProcess();
}
 
void PhysicsList::AddPhysicsList(const G4String& name)
{
  G4cout << "PhysicsList::AddPhysicsList: " << name << " (Ignored)" << G4endl;
}
 