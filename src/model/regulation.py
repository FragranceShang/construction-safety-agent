from pydantic import BaseModel
from datetime import datetime

class Regulation(BaseModel):
    """
    条例元数据
    
    用于表示和管理法规条例的标准化数据结构，包含条例的基本元数据、
    内容信息以及生命周期时间戳。
    
    Attributes:
        regulation_id (str): 具体条文编号，格式为"X.X.X"或"第XX条"
        content (str): 条例详细内容文本，支持多段落格式
        article (str): 条例完整名称，如《中华人民共和国劳动合同法》
        create_time (datetime): 记录创建时间，自动设置为当前时间
        update_time (datetime): 最后更新时间，修改内容时自动更新
        
    Examples:
        >>> regulation = Regulation(
        ...     regulation_id="6.0.6",
        ...     content="用人单位与劳动者建立劳动关系...",
        ...     article="劳动合同法",
        ...     create_time=datetime.now(),
        ...     update_time=datetime.now()
        ... )
    """
    regulation_id: str
    """具体条文编号，格式为"第XX条"或"第XX条第X款" """
    content: str
    """条例详细内容文本，支持多段落格式"""
    article: str
    """条例完整名称，如《中华人民共和国劳动合同法》"""
    create_time: datetime
    """记录创建时间，自动设置为当前时间"""
    update_time: datetime
    """最后更新时间，修改内容时自动更新"""
