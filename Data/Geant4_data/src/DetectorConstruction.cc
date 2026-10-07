#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"

#include "G4Material.hh"
#include "G4Element.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4RunManager.hh"
#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"
#include "G4Exception.hh"

#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <iomanip>

DetectorConstruction::DetectorConstruction()
: G4VUserDetectorConstruction(),
  fLogicTarget(nullptr), fLogicWorld(nullptr),
  fTargetMaterial(nullptr), fWorldMaterial(nullptr),
  fTargetLength(200*m), fTargetRadius(10*m), // Default values
  fMessenger(nullptr)
{
  fMessenger = new DetectorMessenger(this);
  DefineMaterials();
}

DetectorConstruction::~DetectorConstruction()
{
  delete fMessenger;
}

void DetectorConstruction::DefineMaterials()
{
  G4NistManager* nist = G4NistManager::Instance();

  G4Element* H  = nist->FindOrBuildElement("H");
  G4Element* O  = nist->FindOrBuildElement("O");
  G4Element* Mg = nist->FindOrBuildElement("Mg");
  G4Element* Ca = nist->FindOrBuildElement("Ca");
  G4Element* C  = nist->FindOrBuildElement("C");

  // --- World material (vacuum) ---
  fWorldMaterial = nist->FindOrBuildMaterial("G4_Galactic");

  // --- Non-mineral reference materials (kept hardcoded) ---
  // Standard rock as in Hadr01: these ARE mass fractions.
  G4Material* StdRock = new G4Material("StdRock", 2.65*g/cm3, 4, kStateSolid);
  StdRock->AddElement(O,  52.*perCent);
  StdRock->AddElement(Ca, 27.*perCent);
  StdRock->AddElement(C,  12.*perCent);
  StdRock->AddElement(Mg,  9.*perCent);

  // Integer atom counts -> AddElement(G4Element*, G4int) overload.
  G4Material* Water = new G4Material("H2O", 1.*g/cm3, 2);
  Water->AddElement(H, 2);
  Water->AddElement(O, 1);

  nist->FindOrBuildMaterial("G4_SiO2");
  nist->FindOrBuildMaterial("G4_AIR");

  // Minerals are defined at runtime with /testhadr/defineMaterial
  // (see DefineMaterial below). No default target: the macro must
  // set one with /testhadr/TargetMat before /run/initialize.
  fTargetMaterial = nullptr;
}

// Spec format: "<name> <density_g_cm3> <El1> <n1> [<El2> <n2> ...]"
// n_i are relative atom counts (stoichiometry); any normalisation and
// non-integer values are fine. They are converted here to mass fractions,
// since G4Material::AddElement(G4Element*, G4double) expects mass fractions.
void DetectorConstruction::DefineMaterial(const G4String& spec)
{
  auto fail = [&spec](const G4String& why) {
    G4ExceptionDescription ed;
    ed << "Cannot define material from \"" << spec << "\": " << why << G4endl
       << "Expected: <name> <density_g_cm3> <El1> <n1> [<El2> <n2> ...]";
    G4Exception("DetectorConstruction::DefineMaterial", "Mat002",
                FatalErrorInArgument, ed);
  };

  std::istringstream iss(spec);
  std::vector<std::string> tok;
  for (std::string t; iss >> t;) tok.push_back(t);

  if (tok.size() < 4 || tok.size() % 2 != 0) {
    fail("wrong number of tokens (" + std::to_string(tok.size()) + ")");
    return;
  }

  const G4String name = tok[0];

  if (G4Material::GetMaterial(name, false)) {
    G4cout << "### Material " << name
           << " already defined, keeping the existing definition." << G4endl;
    return;
  }

  G4double density = 0.;
  try {
    density = std::stod(tok[1]);
  } catch (const std::exception&) {
    fail("density '" + tok[1] + "' is not a number");
    return;
  }
  if (!(density > 0.)) { fail("density must be > 0"); return; }

  G4NistManager* nist = G4NistManager::Instance();
  std::vector<std::pair<G4Element*, G4double>> comps;  // (element, atom count)

  for (std::size_t i = 2; i < tok.size(); i += 2) {
    G4Element* el = nist->FindOrBuildElement(tok[i]);
    if (!el) { fail("unknown element symbol '" + tok[i] + "'"); return; }

    G4double n = 0.;
    try {
      n = std::stod(tok[i+1]);
    } catch (const std::exception&) {
      fail("count '" + tok[i+1] + "' is not a number");
      return;
    }
    if (!(n > 0.)) { fail("count for " + tok[i] + " must be > 0"); return; }

    // Merge repeated elements
    bool merged = false;
    for (auto& c : comps) {
      if (c.first == el) { c.second += n; merged = true; break; }
    }
    if (!merged) comps.emplace_back(el, n);
  }

  G4double molarMass = 0.;
  for (const auto& c : comps) molarMass += c.second * c.first->GetA();

  auto* mat = new G4Material(name, density*g/cm3,
                             static_cast<G4int>(comps.size()), kStateSolid);

  G4cout << "### Defined material " << name << ", density "
         << density << " g/cm3, mass fractions:";
  for (const auto& c : comps) {
    const G4double w = c.second * c.first->GetA() / molarMass;
    mat->AddElement(c.first, w);
    G4cout << " " << c.first->GetSymbol() << "=" << std::setprecision(4) << w;
  }
  G4cout << G4endl;
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  return ConstructVolumes();
}

G4VPhysicalVolume* DetectorConstruction::ConstructVolumes()
{
  if (!fTargetMaterial) {
    G4Exception("DetectorConstruction::ConstructVolumes", "Mat003",
                FatalException,
                "No target material set. Use /testhadr/TargetMat <name> "
                "before /run/initialize.");
    return nullptr;
  }

  // Cleanup old geometry
  G4GeometryManager::GetInstance()->OpenGeometry();
  G4PhysicalVolumeStore::GetInstance()->Clean();
  G4LogicalVolumeStore::GetInstance()->Clean();
  G4SolidStore::GetInstance()->Clean();

  // --- World ---
  G4double worldSize = 2.0 * std::max(fTargetLength, fTargetRadius) * 1.2;

  G4Box* solidWorld = new G4Box("World", worldSize/2, worldSize/2, worldSize/2);
  fLogicWorld = new G4LogicalVolume(solidWorld, fWorldMaterial, "World");
  G4VPhysicalVolume* physWorld =
    new G4PVPlacement(0, G4ThreeVector(), fLogicWorld, "World", 0, false, 0);

  // --- Target (Cylinder) ---
  G4Tubs* solidTarget = new G4Tubs("Target", 0., fTargetRadius, fTargetLength/2, 0., 360.*deg);
  fLogicTarget = new G4LogicalVolume(solidTarget, fTargetMaterial, "Target");
  new G4PVPlacement(0, G4ThreeVector(), fLogicTarget, "Target", fLogicWorld, false, 0);

  return physWorld;
}

void DetectorConstruction::SetTargetMaterial(const G4String& matName)
{
  // Runtime/custom materials first, then NIST
  G4Material* mat = G4Material::GetMaterial(matName, false);
  if (!mat) mat = G4NistManager::Instance()->FindOrBuildMaterial(matName);

  if (!mat) {
    G4ExceptionDescription ed;
    ed << "Material " << matName << " not found. Define it first with "
       << "/testhadr/defineMaterial (e.g. via /control/execute materials.mac).";
    G4Exception("DetectorConstruction::SetTargetMaterial", "Mat001",
                FatalErrorInArgument, ed);
    return;
  }

  if (mat == fTargetMaterial) {
    G4cout << "### Target material already is set to " << matName << G4endl;
    return;
  }

  fTargetMaterial = mat;
  if (fLogicTarget) fLogicTarget->SetMaterial(fTargetMaterial);
  G4RunManager::GetRunManager()->PhysicsHasBeenModified();
  G4cout << "### Target material set to " << matName << G4endl;
}

void DetectorConstruction::SetTargetRadius(G4double val)
{
  fTargetRadius = val;
  G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::SetTargetLength(G4double val)
{
  fTargetLength = val;
  G4RunManager::GetRunManager()->ReinitializeGeometry();
}
