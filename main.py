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

# 设定的起始日期 (2025年1月1日)
START_DATE = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
# 每封邮件包含的论文数量
BATCH_SIZE = 20

def fetch_arxiv_papers():
    """从 arXiv 获取 2025 年至今的所有相关论文"""
    print(f"开始检索 arXiv (自 {START_DATE.date()} 起)...")
    papers =[]
    
    for query in QUERIES:
        print(f"正在检索关键词: {query}")
        # 配置 arxiv 客户端，加入延迟防封
        client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
        search = arxiv.Search(
            query=query,
            sort_by=arxiv.SortCriterion.SubmittedDate # 按时间倒序
        )
        
        try:
            for result in client.results(search):
                # 如果论文发布时间早于起始日期，则该关键词停止检索（因为是倒序的）
                if result.published < START_DATE:
                    break
                    
                papers.append({
                    "title": result.title,
                    "url": result.pdf_url,
                    "abstract": result.summary.replace("\n", " "),
                    "source": "arXiv"
                })
        except Exception as e:
            print(f"arXiv 检索中断: {e}")
            
    return papers

def fetch_semantic_scholar():
    """从 Semantic Scholar 获取顶级会议/期刊论文"""
    print("开始检索 Semantic Scholar (顶会/顶刊)...")
    papers =[]
    url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    
    # 将多个关键词组合进行宽泛搜索
    query_str = "deepfake detection OR text-to-video OR watermarking"
    
    params = {
        'query': query_str,
        'fields': 'title,url,abstract,venue,publicationDate',
        'publicationTypes': 'JournalArticle,Conference',
        'year': '2025-2026', # 限制 2025 至今
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
                    papers.append({
                        "title": item['title'],
                        "url": item['url'] if item.get('url') else "无链接",
                        "abstract": item['abstract'],
                        "source": f"Top Venue - {venue}"
                    })
    except Exception as e:
        print(f"Semantic Scholar 检索出错: {e}")
        
    return papers

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
        response = client.chat.completions.create(
            model="qwen3-vl-plus-2025-12-19", # 换成用户指定模型
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

if __name__ == "__main__":
    # 1. 收集论文
    all_papers = fetch_arxiv_papers() + fetch_semantic_scholar()
    
    # 2. 标题去重
    seen_titles = set()
    unique_papers =[]
    for p in all_papers:
        title_lower = p['title'].lower()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_papers.append(p)
            
    total_papers = len(unique_papers)
    print(f"🎉 检索完毕！自 2025 年至今共发现 {total_papers} 篇不重复的相关论文。")
    
    if total_papers == 0:
        print("无相关论文。")
        exit()
        
    # 计算需要分几封邮件
    total_batches = (total_papers + BATCH_SIZE - 1) // BATCH_SIZE

    # 3. 分批处理与发送
    for i in range(0, total_papers, BATCH_SIZE):
        batch = unique_papers[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        print(f"\n--- 正在处理第 {batch_num}/{total_batches} 批次 (共 {len(batch)} 篇) ---")
        
        batch_report = ""
        for local_idx, paper in enumerate(batch):
            global_idx = i + local_idx + 1 # 生成全局序号 [1], [2], [3]...
            print(f"  > 提取大模型摘要: [{global_idx}/{total_papers}]")
            
            analysis = analyze_with_llm(paper, global_idx)
            batch_report += analysis + "\n\n" + "—" * 50 + "\n\n"
            
            # API 限流保护：处理完每篇论文休息 3 秒，防止被阿里云封禁
            time.sleep(3) 
            
        # 4. 发送本批次邮件
        send_email(batch_report, batch_num, total_batches)
        
        # 邮箱防屏蔽保护：两封邮件之间强制休息 30 秒，防止被 Gmail 判定为垃圾邮件
        if batch_num < total_batches:
            print("等待 30 秒后发送下一批，防止邮箱风控...")
            time.sleep(30)