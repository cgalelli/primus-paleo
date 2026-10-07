#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4LogicalVolume;
class G4Material;
class DetectorMessenger;

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
  DetectorConstruction();
  virtual ~DetectorConstruction();

public:
  virtual G4VPhysicalVolume* Construct();

  // UI Command setters
  void SetTargetMaterial (const G4String&);
  void SetTargetRadius   (G4double);
  void SetTargetLength   (G4double);
  void DefineMaterial(const G4String& spec);

  // Getters
  G4Material* GetTargetMaterial()  const {return fTargetMaterial;};
  G4double    GetTargetLength()    const {return fTargetLength;};
  G4double    GetTargetRadius()    const {return fTargetRadius;};


private:
  void DefineMaterials();
  G4VPhysicalVolume* ConstructVolumes();

  G4LogicalVolume* fLogicTarget;
  G4LogicalVolume* fLogicWorld;
  G4Material* fTargetMaterial;
  G4Material* fWorldMaterial;
  
  G4double           fTargetLength;
  G4double           fTargetRadius;

  DetectorMessenger* fMessenger;
};

#endif