#include "EventAction.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"

EventAction::EventAction()
: G4UserEventAction(), fPrintModulo(1000)
{}

EventAction::~EventAction()
{}

void EventAction::BeginOfEventAction(const G4Event* evt)
{
  G4int event_id = evt->GetEventID();
  // Print progress every N events
  if (event_id % fPrintModulo == 0) { 
    G4cout << "--> Begin of event: " << event_id << G4endl;
  }
}

void EventAction::EndOfEventAction(const G4Event*)
{}