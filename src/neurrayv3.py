from dataclasses import dataclass
from typing import Literal, TypeAlias



class MatchBase:
    def __init__(self, match_size:ArraySize, emit_size:ArraySize):
        # 3d arrays :despair:
        # axises for: 64+ bit input (ArraySize.count), inverse (2), batch size (1)
        self.match: Neurray = create_neurray(match_size)
        self.emit: Neurray = create_neurray(emit_size)
