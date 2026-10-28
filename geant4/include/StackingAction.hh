#ifndef StackingAction_h
#define StackingAction_h 1

#include "G4UserStackingAction.hh"
#include "globals.hh"

class G4Track;
class RunAction; // Forward declaration

class StackingAction : public G4UserStackingAction
{
  public:
    // The constructor now takes RunAction*
    StackingAction();
    virtual ~StackingAction();

  public:
    virtual G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track*);

};

#endif