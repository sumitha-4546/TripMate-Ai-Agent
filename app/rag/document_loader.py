import re
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from app.utils.logger import logger

KNOWN_DESTINATIONS = {
    "tokyo": {"city": "Tokyo", "country": "Japan"},
    "bangkok": {"city": "Bangkok", "country": "Thailand"},
    "barcelona": {"city": "Barcelona", "country": "Spain"},
    "reykjavik": {"city": "Reykjavik", "country": "Iceland"},
}

SECTIONS = [
    "VISA & ENTRY",
    "BEST TIME TO VISIT",
    "LOCAL CUSTOMS",
    "PACKING TIPS",
    "SAFETY & HEALTH",
]

DEFAULT_DISCLAIMER = (
    "Note: This is a simplified reference document created for a technical assessment exercise. "
    "It reflects general, commonly-known travel patterns and is not authoritative, up-to-date travel or visa advice."
)


def load_destination_documents(data_dir: Path) -> List[Document]:
    """Load and parse destination text files into structured LangChain Document chunks.

    Splits each document into distinct sections preserving city, country, section,
    and disclaimer metadata.
    """
    documents: List[Document] = []

    if not data_dir.exists():
        logger.warning(f"Destination directory does not exist: {data_dir}")
        return documents

    txt_files = list(data_dir.glob("*.txt"))
    # Exclude README.txt from chunking
    txt_files = [f for f in txt_files if f.name.lower() != "readme.txt"]

    logger.info(f"Found {len(txt_files)} destination files to ingest from {data_dir}")

    for file_path in txt_files:
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            city_name = file_path.stem.lower()

            # Identify destination metadata
            dest_meta = KNOWN_DESTINATIONS.get(
                city_name,
                {"city": file_path.stem.capitalize(), "country": "Unknown"},
            )

            city = dest_meta["city"]
            country = dest_meta["country"]

            # Parse disclaimer if present
            disclaimer = DEFAULT_DISCLAIMER
            disclaimer_match = re.search(r"Note:\s*(.*?)(?=\n[A-Z &]{3,}|\Z)", content, re.DOTALL)
            if disclaimer_match:
                disclaimer = disclaimer_match.group(0).strip().replace("\n", " ")

            # Parse sections: matches section headers like VISA & ENTRY, BEST TIME TO VISIT, etc.
            # Pattern looks for known section headers
            section_pattern = r"(VISA & ENTRY|BEST TIME TO VISIT|LOCAL CUSTOMS|PACKING TIPS|SAFETY & HEALTH)"
            splits = re.split(section_pattern, content)

            # splits[0] is preamble / header, then alternating (section_name, section_body)
            if len(splits) > 1:
                for i in range(1, len(splits), 2):
                    sec_name = splits[i].strip()
                    sec_body = splits[i + 1].strip() if i + 1 < len(splits) else ""

                    if sec_body:
                        # Rich text content including context
                        chunk_text = (
                            f"Destination: {city}, {country}\n"
                            f"Section: {sec_name}\n\n"
                            f"{sec_body}\n\n"
                            f"[{disclaimer}]"
                        )

                        doc = Document(
                            page_content=chunk_text,
                            metadata={
                                "city": city,
                                "country": country,
                                "section": sec_name,
                                "source": file_path.name,
                                "disclaimer": disclaimer,
                            },
                        )
                        documents.append(doc)
            else:
                # Fallback if section headers format differs
                doc = Document(
                    page_content=content,
                    metadata={
                        "city": city,
                        "country": country,
                        "section": "General Guide",
                        "source": file_path.name,
                        "disclaimer": disclaimer,
                    },
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error parsing destination document {file_path.name}: {e}")

    logger.info(f"Successfully parsed {len(documents)} document chunks across destinations.")
    return documents

