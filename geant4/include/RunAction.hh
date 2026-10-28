#ifndef RunAction_h
#define RunAction_h 1

#include "G4UserRunAction.hh"
#include "globals.hh"
#include <fstream>

class G4Run;

class RunAction : public G4UserRunAction
{
  public:
    RunAction();
    virtual ~RunAction();

    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);

    // Helper method used by StackingAction to write to the file
    std::ofstream* GetOutputFile() const;

  private:
    std::ofstream* fOutputFile;
};

#endif