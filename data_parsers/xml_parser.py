"""
CTI XML Dataset Parser

Discovers, parses, and converts CTI XML dataset files into structured
event dictionaries with human-readable narratives suitable for LLM
consumption.

Handles both ReportEvent and MalwareEvent XML formats from the
CTIMiner dataset collection (2008-2019).
"""

import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CTIXMLParser:
    """
    Parses CTI XML dataset files and produces structured event records.

    Each event is converted to a dictionary containing:
      - event_id: Unique event ID from the XML
      - file_source: Source XML filename
      - date: Event date (YYYY-MM-DD)
      - info: Event info/description field
      - narrative: Human-readable text narrative for LLM processing
      - attributes: Raw structured attribute data for downstream use

    Usage:
        parser = CTIXMLParser("path/to/CTI_Report_Dataset")
        events = parser.parse_all()
        for event in events:
            print(event["narrative"])
    """

    def __init__(self, dataset_dir: str) -> None:
        """
        Initialize the parser with a dataset directory path.

        Args:
            dataset_dir: Path to the directory containing XML files.

        Raises:
            FileNotFoundError: If the dataset directory does not exist.
        """
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        if not self.dataset_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.dataset_dir}")

        logger.info("CTIXMLParser initialized with directory: %s", self.dataset_dir)

    def discover_files(self) -> List[Path]:
        """
        Discover all XML files in the dataset directory.

        Returns:
            Sorted list of Path objects for all .xml files found.
        """
        xml_files = sorted(self.dataset_dir.glob("*.xml"))
        logger.info("Discovered %d XML files", len(xml_files))
        return xml_files

    def parse_file(self, filepath: Path) -> List[Dict]:
        """
        Parse a single XML file and extract all events.

        Args:
            filepath: Path to the XML file.

        Returns:
            List of event dictionaries. Empty list if the file has no events
            (e.g., empty datasets like <CTIMinerDataset/>).
        """
        filepath = Path(filepath)
        filename = filepath.name
        events = []

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error("XML parse error in %s: %s", filename, e)
            return []

        # Handle empty datasets (<CTIMinerDataset/>)
        event_elements = root.findall("Event")
        if not event_elements:
            logger.debug("No events found in %s (empty dataset)", filename)
            return []

        for event_elem in event_elements:
            try:
                event_data = self._build_event_record(event_elem, filename)
                events.append(event_data)
            except Exception as e:
                event_id = event_elem.findtext("id", "unknown")
                logger.warning(
                    "Error parsing event %s in %s: %s", event_id, filename, e
                )
                continue

        logger.info("Parsed %d events from %s", len(events), filename)
        return events

    def parse_all(self) -> List[Dict]:
        """
        Parse all XML files in the dataset directory.

        Returns:
            Combined list of all event dictionaries from all files,
            sorted by file source then event ID.
        """
        all_events = []
        xml_files = self.discover_files()

        if not xml_files:
            logger.warning("No XML files found in %s", self.dataset_dir)
            return []

        for filepath in xml_files:
            file_events = self.parse_file(filepath)
            all_events.extend(file_events)

        logger.info(
            "Total: parsed %d events from %d files",
            len(all_events),
            len(xml_files),
        )
        return all_events

    def _build_event_record(self, event_elem: ET.Element, filename: str) -> Dict:
        """
        Build a structured event record from an XML Event element.

        Args:
            event_elem: The <Event> XML element.
            filename: Source filename for provenance tracking.

        Returns:
            Dictionary with event_id, file_source, date, info,
            narrative, and attributes fields.
        """
        event_id = event_elem.findtext("id", "").strip()
        date = event_elem.findtext("date", "").strip()
        info = event_elem.findtext("info", "").strip()
        
        # Clean filename for cleaner ID (e.g., CTIMiner_APT_1.xml -> CTIMiner_APT_1)
        clean_filename = filename.replace(".xml", "")
        global_id = f"{clean_filename}_{event_id}"

        # Parse all attribute items
        attributes = self._parse_attributes(event_elem)

        # Build the narrative text
        narrative = self._build_narrative(
            event_id=event_id,
            date=date,
            info=info,
            filename=filename,
            attributes=attributes,
        )

        return {
            "global_id": global_id,
            "event_id": event_id,
            "file_source": filename,
            "date": date,
            "info": info,
            "narrative": narrative,
            "attributes": attributes,
        }

    def _parse_attributes(self, event_elem: ET.Element) -> List[Dict]:
        """
        Parse all <item> elements within an Event's <Attribute> block.

        Args:
            event_elem: The <Event> XML element.

        Returns:
            List of attribute dictionaries, each with:
            category, type, value, comment, id.
        """
        attributes = []
        attr_block = event_elem.find("Attribute")

        if attr_block is None:
            return attributes

        for item in attr_block.findall("item"):
            attr = {
                "category": item.findtext("category", "").strip(),
                "type": item.findtext("type", "").strip(),
                "value": item.findtext("value", "").strip(),
                "comment": (item.findtext("comment") or "").strip(),
                "id": item.findtext("id", "").strip(),
            }
            attributes.append(attr)

        return attributes

    def _build_narrative(
        self,
        event_id: str,
        date: str,
        info: str,
        filename: str,
        attributes: List[Dict],
    ) -> str:
        """
        Build a human-readable narrative text from event data.

        Groups attributes by category and formats them into a readable
        text block suitable for LLM processing.

        Args:
            event_id: The event's unique identifier.
            date: Event date string.
            info: Event info/description.
            filename: Source XML filename.
            attributes: Parsed attribute list.

        Returns:
            Formatted narrative string.
        """
        # Header
        lines = [
            f"Event {event_id} from {filename} occurred on {date}.",
        ]

        if info:
            lines.append(f"Info: {info}")

        lines.append("")  # blank line separator

        if not attributes:
            lines.append("No attributes recorded for this event.")
            return "\n".join(lines)

        # Group attributes by category
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for attr in attributes:
            grouped[attr["category"]].append(attr)

        # Format each category group
        for category, items in grouped.items():
            lines.append(f"[{category}]")

            for item in items:
                value = item["value"]
                attr_type = item["type"]
                comment = item["comment"]

                if comment:
                    lines.append(f"  - {attr_type}: {value} ({comment})")
                else:
                    lines.append(f"  - {attr_type}: {value}")

            lines.append("")  # blank line between categories

        return "\n".join(lines).rstrip()

    def get_summary(self) -> Dict:
        """
        Generate a summary of the dataset without full parsing.

        Returns:
            Dictionary with file_count, files (list of {name, event_count}),
            and total_events.
        """
        xml_files = self.discover_files()
        summary = {
            "file_count": len(xml_files),
            "files": [],
            "total_events": 0,
        }

        for filepath in xml_files:
            try:
                tree = ET.parse(filepath)
                root = tree.getroot()
                event_count = len(root.findall("Event"))
            except ET.ParseError:
                event_count = 0

            summary["files"].append({
                "name": filepath.name,
                "event_count": event_count,
            })
            summary["total_events"] += event_count

        return summary
