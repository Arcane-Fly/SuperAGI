from abc import ABC
from superagi.tools.base_tool import BaseToolKit, BaseTool
from typing import Type, List
from superagi.tools.google_calendar.create_calendar_event import CreateEventCalendarTool
from superagi.tools.google_calendar.delete_calendar_event import DeleteCalendarEventTool
from superagi.tools.google_calendar.list_calendar_events import ListCalendarEventsTool
from superagi.tools.google_calendar.event_details_calendar import EventDetailsCalendarTool


class GoogleCalendarToolKit(BaseToolKit, ABC):
    name: str = "Email Toolkit"
    description: str = "Email Tool kit contains all tools related to sending email"

    def get_tools(self) -> List[BaseTool]:
        return [CreateEventCalendarTool(), DeleteCalendarEventTool(), ListCalendarEventsTool(), EventDetailsCalendarTool()]

    def get_env_keys(self) -> List[str]:
        return ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"]