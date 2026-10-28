#ifndef PrimaryGeneratorAction_h
#define PrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4ParticleGun;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
  PrimaryGeneratorAction();
  virtual ~PrimaryGeneratorAction();

  virtual void GeneratePrimaries(G4Event*);

private:

  PrimaryGeneratorAction & operator=(const PrimaryGeneratorAction &right);
  PrimaryGeneratorAction(const PrimaryGeneratorAction&);

  G4ParticleGun*   fParticleGun;
};

#endif


