from pydantic import BaseModel

class NetworkEvent(BaseModel):
    Duration: float
    Src_Pt: int
    Dst_Pt: int
    Bytes: float
    bytes_per_packet: float
