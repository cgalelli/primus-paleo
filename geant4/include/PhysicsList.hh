#ifndef PhysicsList_h
#define PhysicsList_h 1

#include "G4VModularPhysicsList.hh"
#include "globals.hh"

class PhysicsList: public G4VModularPhysicsList
{
public:
  PhysicsList();
  virtual ~PhysicsList();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

  // Helper to add physics constructors if needed
  void AddPhysicsList(const G4String& name);

private:
  std::vector<G4String> fHadronPhys;
};

#endif