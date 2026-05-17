from __future__ import annotations
import math
from typing import Dict, Literal, Optional, List, Annotated

from pydantic import Field, BaseModel

                           
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from method.environment import Environment
    from method.agent.persona import Persona


class Content(BaseModel):
    """
    一个简单的内存内容存储类。
    在整个ReAct周期中，智能体通过工具与这个类的实例交互，
    以存储和检索运行中产生的内容。
    """
    id: str = Field(..., description="一个全局唯一的标识符，确保每条内容都可追溯。")
    author_id: str = Field(..., description="发布该内容的智能体id，用于关联智能体的行为和属性。")
    time: int = Field(..., description="内容的发布时间步 t，用于计算时效性和记忆衰减。")
    content_type: Literal['image', 'video'] = Field(..., description="内容的类型，可以是图片或视频。")
    topic: str = Field(..., description="内容的主题，用于分类和搜索。")
    content_detail: str = Field(..., description="内容的详细描述")
    reason: str = Field(..., description="发布这个内容的理由")

    watermark_id: Optional[str] = Field(description="内容水印id，用于标识内容来源。")
    ai_proportion: float = Field(default=0.0, description="内容的客观AI生成比例")

    views: int = Field(default=0, description="内容的浏览次数，用于评估内容的质量。")
    likes: int = Field(default=0, description="内容的点赞数，用于评估内容受欢迎程度。")
    shares: int = Field(default=0, description="内容的分享数，用于评估内容传播效果。")
    comments: Optional[List[Dict[str, str]]] = Field(..., description="内容的评论，用于评估内容互动性。 评论者：内容")

    evasion: Optional[str] = Field(..., description="内容所遭受的攻击手段")
    platform_label: Optional[Literal['AI', 'HUMAN']] = Field(...,
                                                             description="内容的标签，平台打标，用于区分 AI 生成的内容与人工生成的内容。")
    true_label: Literal['AI', 'HUMAN'] = Field(..., description="内容的真实标签。")


class ContentStore:

    def __init__(self):
        self._data: Dict[str, Content] = {}        

    async def add_content(self, content: Content, environment: Environment) -> bool:
        """
        添加内容
        :param environment:
        :param content:
        :return:
        """
        try:
                  
            if content.id in self._data.keys():
                return False
            self._data[content.id] = content

                     
            if environment and environment.memories_store:
                                           
                await environment.memories_store.add_content_to_db(content)
        except:
            return False

        return True

    def add_content_list(self, content_list: List[Content], environment: Environment) -> bool:
        """
        批量添加内容
        :param environment:
        :param content_list:
        :return:
        """
        for content in content_list:
            if not self.add_content(content, environment):
                return False
        return True

    def get_content_by_id(
            self,
            content_id: str,
            viewer_persona: Optional["Persona"] = None,
    ) -> Content | None:
        """
        通过id获取内容
        :param content_id:
        :param viewer_persona: 当前查看该内容的智能体（预留参数，便于上层根据查看者做同步过滤）
        :return:
        """
        try:
            content = self._data[content_id]
                                                                    

        except:
            return None

                                        
                                                    
        return content

    async def get_content_by_limit_return_str(self, limit: int, persona: Persona, interest_content: str,
                                              environment: Environment) -> str:
        """
        获取内容
        :param interest_content: 兴趣点
        :param environment:
        :param persona:
        :param limit:
        :return:
        """

                    
        content_ids = await environment.memories_store.recommend_contents(persona, interest_content,
                                                                          environment.day_time,
                                                                          limit)

        if len(content_ids) == 0:
            return "今日，平台上的最新内容，你已经浏览完了"

        contents = []
        for content_id in content_ids:
            content = self.get_content_by_id(content_id)
            self.update_content_views_by_id(content_id)
            if content:
                contents.append(content)

              
        return_content = ""
        for content in contents:
            return_content += f"""
        ---
        关于内容的信息如下：
        内容唯一标识符：{content.id}
        内容发布者：{content.author_id}
        内容类型：{content.content_type}
        内容主题：{content.topic}
        内容摘要：{content.content_detail[:20]} + ....
        内容标签：{content.platform_label}
        ---
        """
        return return_content

    def update_content_likes_by_id(self, content_id: str) -> bool:
        """
        更新内容点赞数
        :param content_id:
        :return:
        """
        try:
            self._data[content_id].likes += 1
        except:
            return False
        return True

    def update_content_shares_by_id(self, content_id: str) -> bool:
        """
        更新内容分享数
        :param content_id:
        :return:
        """
        try:
            self._data[content_id].shares += 1
        except:
            return False
        return True

    def update_content_views_by_id(self, content_id: str) -> bool:
        """
        增加内容浏览量
        :param content_id:
        :return:
        """
        try:
            self._data[content_id].views += 1
        except:
            return False
        return True

    def update_content_comments_by_id(self, content_id: str, persona_id: str, comment: str) -> bool:
        """
        新增评论内容
        :param content_id: 评论的内容id
        :param persona_id: 公众id
        :param comment: 评论内容
        :return:
        """
        try:
            self._data[content_id].comments.append({persona_id: comment})
        except:
            return False
        return True

    def search_by_topic_return_str(self, topic: str, limit: int = 10) -> str:

        contents = [content for k, content in self._data.items() if content.topic == topic]
        for content in contents:
            self.update_content_views_by_id(content_id=content.id)

              
        return_content = ""
        for content in contents:
            return_content += f"""
        ---
        关于内容{content.id}的信息如下：
        内容唯一标识符：{content.id}
        内容发布者：{content.author_id}
        内容类型：{content.content_type}
        内容主题：{content.topic}
        内容标签：{content.platform_label}
        ---
        """
        return return_content

    def get_end_content_id(self) -> int:
        """
        获取最后一条内容id
        :return:
        """
        max_id = 0

        for content_id in self._data.keys():
            if content_id.isdigit():
                val = int(content_id)
                if val > max_id:
                    max_id = val

        return max_id

    def get_contents_by_author_id(self, author_id: str) -> list:
        """
        根据作者id获取内容
        :param author_id:
        :return:
        """
        contents = [v for k, v in self._data.items() if v.author_id == author_id]
        for content in contents:
            self.update_content_views_by_id(content_id=content.id)
        return contents

    def get_all_contents(self) -> list:
        """
        获取所有内容
        :return:
        """
        return list(self._data.values())

    def get_all_contents_dict(self) -> list:
        """
        获取所有内容
        :return:
        """
        return list(self._data.values())

    def get_all_content_ids(self) -> list:
        """
        获取所有内容
        :return:
        """
        return list(self._data.keys())

    def calculate_content_influence(self, content: Content, environment, initial_score: bool = False) -> float:
        """
        计算内容对智能体的影响（及内容价值）。
        """
        current_day = environment.day_time

                             
                                       
        standard_x0 = calculate_dynamic_x0(environment)

                            
        influence_weights = {
            'views': 0.1,            
            'likes': 0.2,               
            'shares': 0.5,               
            'comments': 0.3               
        }

                        
        time_decay_halflife_days = 2.0

                                    
        views = content.views
        likes = content.likes
        shares = content.shares
        comments = content.comments
        comments_count = len(comments)

        raw_influence = (
                views * influence_weights['views'] +
                likes * influence_weights['likes'] +
                shares * influence_weights['shares'] +
                comments_count * influence_weights['comments']
        )

                             
        content_day = content.time
        days_passed = max(0, current_day - content_day)

                
        decay_factor = 0.5 ** (days_passed / time_decay_halflife_days)

                                          
        final_influence_score = raw_influence * decay_factor

                                   
        if initial_score:
            return final_influence_score

                                                                   
                                
                                                                   

                  
                                                         
                          

        if days_passed < 1.0:
                                     
                                                   
            dynamic_x0 = standard_x0 * 0.1
        elif days_passed < 3.0:
                                    
            dynamic_x0 = standard_x0 * 0.5
        else:
                            
            dynamic_x0 = standard_x0

                   
        k = 0.05

                            
        try:
                                   
            sigmoid_value = 1 / (1 + math.exp(-k * (final_influence_score - dynamic_x0)))

                     
                                
                                                  
                                                        
                                               
            value = sigmoid_value
        except OverflowError:
            value = 0.1             

        return value


def calculate_dynamic_x0(environment):
    """
    根据当前环境的规模，动态计算影响力价值中点 x0。
    """
                           
    total_audience = environment.initial_persona_count

    if total_audience == 0:
        return 1.0                    

                                
                      

                           
    estimated_views = total_audience * 1.0

                      
    estimated_likes = estimated_views * 0.6

                      
    estimated_comments = estimated_likes * 0.6

                      
    estimated_shares = estimated_likes * 0.2

                                                  
    influence_weights = {
        'views': 0.1,            
        'likes': 0.2,               
        'shares': 0.5,               
        'comments': 0.3               
    }

    theoretical_raw_influence = (
            estimated_views * influence_weights['views'] +
            estimated_likes * influence_weights['likes'] +
            estimated_shares * influence_weights['shares'] +
            estimated_comments * influence_weights['comments']
    )

                                
                      
    return max(0.5, theoretical_raw_influence)
