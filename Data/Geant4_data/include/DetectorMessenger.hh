#ifndef DetectorMessenger_h
#define DetectorMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

class DetectorConstruction;
class G4UIcmdWithAString;
class G4UIcmdWithADoubleAndUnit;

class DetectorMessenger: public G4UImessenger
{
public:
  DetectorMessenger(DetectorConstruction*);
  virtual ~DetectorMessenger();

  virtual void SetNewValue(G4UIcommand*, G4String);

private:
  DetectorConstruction* fDetector;
  G4UIdirectory* fTestDir;
  G4UIcmdWithAString* fMatCmd;
  G4UIcmdWithADoubleAndUnit* fRCmd;
  G4UIcmdWithADoubleAndUnit* fLCmd;
  G4UIcmdWithAString* fDefMatCmd;
};

#endif