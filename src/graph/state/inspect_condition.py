from model.inspection_state import InspectionState


def should_execute_followup(state: InspectionState) -> str:
    plans = state.get("followup_plan", [])
    actionable = [item for item in plans if item.get("observability") == "same_image_recoverable"]
    return "execute" if actionable else "skip"
