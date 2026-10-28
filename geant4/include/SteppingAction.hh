#ifndef SteppingAction_h
#define SteppingAction_h 1

#include "G4UserSteppingAction.hh"
#include "globals.hh"

class RunAction;

class SteppingAction : public G4UserSteppingAction
{
  public:
    SteppingAction(RunAction* runAction);
    virtual ~SteppingAction();

    virtual void UserSteppingAction(const G4Step*);

  private:
    RunAction* fRunAction;
};

#endif