from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet


@dataclass(frozen=True)
class SheetSpec:
    name: str
    paths: List[str]


class SampleWorkbook:
    def __init__(self, path: Path, specs: Sequence[SheetSpec], headers: List[str]) -> None:
        self.path = path
        self.headers = headers
        self.specs = list(specs)
        if path.exists():
            self.workbook = load_workbook(path)
        else:
            self.workbook = Workbook()
            if self.specs:
                self.workbook.active.title = self.specs[0].name
        self.sheet_entries: Dict[str, set[str]] = {}
        for index, spec in enumerate(self.specs):
            sheet = self._ensure_sheet(spec.name, index == 0)
            self._ensure_headers(sheet)
            existing: set[str] = set()
            for row in sheet.iter_rows(min_row=2, max_col=1, values_only=True):
                value = row[0]
                if value:
                    existing.add(str(value))
            self.sheet_entries[spec.name] = existing
        if not self.specs and "Sheet" in self.workbook.sheetnames:
            sheet = self.workbook["Sheet"]
            if sheet.max_row == 0:
                sheet.append(self.headers)
            self.sheet_entries[sheet.title] = set()

    def _ensure_sheet(self, name: str, first: bool) -> Worksheet:
        if name in self.workbook.sheetnames:
            return self.workbook[name]
        if first and len(self.workbook.sheetnames) == 1 and self.workbook.active.max_row <= 1:
            sheet = self.workbook.active
            sheet.title = name
            return sheet
        return self.workbook.create_sheet(title=name)

    def _ensure_headers(self, sheet: Worksheet) -> None:
        if sheet.max_row == 0:
            sheet.append(self.headers)
            return
        first_row = [cell.value for cell in sheet[1][: len(self.headers)]]
        if first_row != self.headers:
            sheet.insert_rows(1)
            for index, header in enumerate(self.headers, start=1):
                sheet.cell(row=1, column=index, value=header)

    def has_entry(self, sheet_name: str, filename: str) -> bool:
        entries = self.sheet_entries.get(sheet_name)
        if entries is None:
            return False
        return filename in entries

    def append_record(self, sheet_name: str, record: Dict[str, object]) -> None:
        sheet = self.workbook[sheet_name]
        row = [self._cell_value(record.get(header)) for header in self.headers]
        sheet.append(row)
        filename = str(record.get("filename", ""))
        if filename:
            self.sheet_entries[sheet_name].add(filename)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.workbook.save(self.path)

    def _cell_value(self, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, float):
            return float(value)
        return value
