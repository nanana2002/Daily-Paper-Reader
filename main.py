import arxiv
import requests
import datetime
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import OpenAI

# 1. 环境变量读取
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")

# ================= 检索配置区 =================
# 核心检索关键词
QUERIES =[
    'all:"image-to-video" OR all:"text-to-video" OR all:"I2V" OR all:"T2V"',
    'all:"deepfake detection" OR all:"multimedia forgery" OR all:"AIGC detection"',
    'all:"proactive defense" OR all:"watermarking" OR all:"adversarial watermarking"'
]

# CCF-A / 顶会顶刊 关键词 (用于匹配 Semantic Scholar 等)
CCF_A_VENUES =["CVPR", "ICCV", "ECCV", "NeurIPS", "ICML", "ICLR", "AAAI", "IJCAI", "ACM Multimedia", "TPAMI", "IJCV"]

# 设定的起始日期搜索近三天的论文
START_DATE = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)

# 每封邮件包含的论文数量
BATCH_SIZE = 20

# 全局变量：记录批次编号和累计论文数
batch_counter = 1
total_papers_processed = 0
total_batches = 0  # 先占位，后续动态计算


def analyze_with_llm(paper, index):
    """调用大模型提炼论文信息（彻底去废话版）"""
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY, 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 极度严厉的 Prompt 限制大模型说废话
    prompt = f"""
    你是一个无情的API格式化机器。请阅读以下论文标题和摘要：
    Title: {paper['title']}
    Abstract: {paper['abstract']}
    
    【强制执行规则】
    1. 绝对不准输出任何开头问候语（例如“我将为您输出...”）。
    2. 绝对不准输出任何结尾说明（例如“注意：链接可能...”）。
    3. 只允许输出以下5行结构化内容，不要有任何多余字符。

    📌 **[{index}] 标题：** [中文标题翻译] ({paper['title']})
    🔗 **链接：** {paper['url']}
    📚 **来源：** {paper['source']}
    📝 **摘要：** [用中文简练总结核心内容，约100字]
    ✨ **亮点：** [一到两句话点明创新点或解决的痛点]
    """
    try:
        # ========== 核心修改：替换模型名称 ==========
        response = client.chat.completions.create(
            model="qwen3-vl-plus-2025-12-19",  # 改为你指定的模型
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, # 降低温度，拒绝自由发挥
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"📌 **[{index}] 标题：** {paper['title']}\n❌ 大模型处理失败: {e}"


def send_email(content, batch_current, batch_total):
    """分开发送批次邮件"""
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = f"📚 AIGC与伪造防御 25年至今汇总 (第 {batch_current}/{batch_total} 批)"
    
    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2 style="color: #2C3E50;">🚀 全量学术汇总 (分批包: {batch_current}/{batch_total})</h2>
        <div style="background-color: #F9F9F9; padding: 15px; border-radius: 8px;">
          {content.replace(chr(10), '<br>')}
        </div>
      </body>
    </html>
    """
    msg.attach(MIMEText(html_content, 'html', 'utf-8'))
    
    try:
        server = smtplib.SMTP_SSL(SMTP_SERVER, 465)
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, [EMAIL_RECEIVER], msg.as_string())
        server.quit()
        print(f"✅ 第 {batch_current}/{batch_total} 批邮件发送成功！")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")


def process_and_send_batch(batch_papers, batch_num, total_batch_num):
    """处理单批次论文并发送邮件"""
    global total_papers_processed
    batch_report = ""
    
    for local_idx, paper in enumerate(batch_papers):
        global_idx = total_papers_processed + local_idx + 1  # 全局序号
        print(f"  > 提取大模型摘要: [{global_idx}]")
        
        analysis = analyze_with_llm(paper, global_idx)
        batch_report += analysis + "\n\n" + "—" * 50 + "\n\n"
        
        # API 限流保护：处理完每篇论文休息 3 秒
        time.sleep(3)
    
    # 发送本批次邮件
    send_email(batch_report, batch_num, total_batch_num)
    total_papers_processed += len(batch_papers)
    
    # 邮箱防屏蔽保护：非最后一批则休息 30 秒
    if batch_num < total_batch_num:
        print("等待 30 秒后处理下一批，防止邮箱风控...")
        time.sleep(30)


def fetch_arxiv_papers_and_send():
    """从 arXiv 边检索边发送论文"""
    global batch_counter, total_batches
    arxiv_papers = []
    print(f"开始检索 arXiv (自 {START_DATE.date()} 起)...")
    
    for query in QUERIES:
        print(f"正在检索关键词: {query}")
        # 配置 arxiv 客户端，加入延迟防封
        client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
        search = arxiv.Search(
            query=query,
            sort_by=arxiv.SortCriterion.SubmittedDate  # 按时间倒序
        )
        
        try:
            for result in client.results(search):
                # 如果论文发布时间早于起始日期，则该关键词停止检索（因为是倒序的）
                if result.published < START_DATE:
                    break
                
                # 收集单篇论文
                arxiv_papers.append({
                    "title": result.title,
                    "url": result.pdf_url,
                    "abstract": result.summary.replace("\n", " "),
                    "source": "arXiv"
                })
                
                # 达到批次大小则立即处理并发送
                if len(arxiv_papers) >= BATCH_SIZE:
                    process_and_send_batch(arxiv_papers, batch_counter, total_batches)
                    batch_counter += 1
                    arxiv_papers = []  # 清空当前批次
                
        except Exception as e:
            print(f"arXiv 检索中断: {e}")
    
    # 处理 arXiv 剩余的不足一批的论文
    return arxiv_papers


def fetch_semantic_scholar_and_send():
    """从 Semantic Scholar 边检索边发送论文"""
    global batch_counter, total_batches
    s2_papers = []
    print("开始检索 Semantic Scholar (顶会/顶刊)...")
    url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    
    # 将多个关键词组合进行宽泛搜索
    query_str = "deepfake detection OR text-to-video OR watermarking"
    
    params = {
        'query': query_str,
        'fields': 'title,url,abstract,venue,publicationDate',
        'publicationTypes': 'JournalArticle,Conference',
        'year': '2025-2026',  # 限制 2025 至今
        'limit': 100
    }
    
    try:
        response = requests.get(url, params=params).json()
        if 'data' in response:
            for item in response['data']:
                venue = item.get('venue', '')
                # 只保留在 CCF-A 列表中的，或 venue 包含我们核心会议的
                is_top_venue = any(top_v.lower() in str(venue).lower() for top_v in CCF_A_VENUES)
                
                if is_top_venue and item.get('abstract'):
                    # 收集单篇论文
                    s2_papers.append({
                        "title": item['title'],
                        "url": item['url'] if item.get('url') else "无链接",
                        "abstract": item['abstract'],
                        "source": f"Top Venue - {venue}"
                    })
                    
                    # 达到批次大小则立即处理并发送
                    if len(s2_papers) >= BATCH_SIZE:
                        process_and_send_batch(s2_papers, batch_counter, total_batches)
                        batch_counter += 1
                        s2_papers = []  # 清空当前批次
                        
    except Exception as e:
        print(f"Semantic Scholar 检索出错: {e}")
    
    # 处理 Semantic Scholar 剩余的不足一批的论文
    return s2_papers


if __name__ == "__main__":
    # 第一步：先预估总批次（先全量检索统计总数，再重置检索）
    # 注意：如果不想预估总批次，可将 total_batches 设为 "未知"，邮件标题改为动态显示
    temp_arxiv = []
    temp_s2 = []
    
    # 临时检索统计总数（仅统计，不处理）
    print("🔍 先统计总论文数，用于批次显示...")
    # 统计 arXiv 数量
    for query in QUERIES:
        client = arxiv.Client(page_size=100, delay_seconds=1, num_retries=2)
        search = arxiv.Search(query=query, sort_by=arxiv.SortCriterion.SubmittedDate)
        try:
            for result in client.results(search):
                if result.published < START_DATE:
                    break
                temp_arxiv.append(1)
        except:
            pass
    
    # 统计 Semantic Scholar 数量
    url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    query_str = "deepfake detection OR text-to-video OR watermarking"
    params = {'query': query_str, 'fields': 'venue,abstract', 'publicationTypes': 'JournalArticle,Conference', 'year': '2025-2026', 'limit': 100}
    try:
        response = requests.get(url, params=params).json()
        if 'data' in response:
            for item in response['data']:
                venue = item.get('venue', '')
                is_top_venue = any(top_v.lower() in str(venue).lower() for top_v in CCF_A_VENUES)
                if is_top_venue and item.get('abstract'):
                    temp_s2.append(1)
    except:
        pass
    
    # 计算总批次
    total_papers = len(temp_arxiv) + len(temp_s2)
    total_batches = (total_papers + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📊 预估总论文数：{total_papers}，总批次：{total_batches}")
    
    # 第二步：正式边搜边发
    print("\n========== 开始边检索边发送 ==========")
    # 处理 arXiv 论文（边搜边发）
    remaining_arxiv = fetch_arxiv_papers_and_send()
    # 处理 Semantic Scholar 论文（边搜边发）
    remaining_s2 = fetch_semantic_scholar_and_send()
    
    # 第三步：处理所有剩余的论文（不足一批的部分）
    remaining_all = remaining_arxiv + remaining_s2
    if remaining_all:
        print(f"\n处理剩余 {len(remaining_all)} 篇论文...")
        process_and_send_batch(remaining_all, batch_counter, total_batches)
    
    print("\n🎉 所有论文检索和发送完成！")
