import arxiv
import requests
import datetime
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import OpenAI

# 1. 环境变量读取
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.qq.com")

# 2. 设置检索关键词 (针对你的专业)
QUERIES =[
    'all:"text-to-video" OR all:"image-to-video"',
    'all:"deepfake detection" OR all:"multimedia forgery"',
    'all:"proactive defense" OR all:"watermarking"'
]

def fetch_arxiv_papers():
    """从 arXiv 获取过去 24 小时内的相关论文"""
    print("开始检索 arXiv 论文...")
    papers =[]
    # 找最近 2 天的数据，防止时差遗漏
    for query in QUERIES:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=5, # 每个关键词最多取5篇，防止太多读不完
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "url": result.pdf_url,
                "abstract": result.summary.replace("\n", " "),
                "source": "arXiv"
            })
    return papers

def fetch_semantic_scholar():
    """从 Semantic Scholar 获取近期发表的会议期刊论文"""
    print("开始检索 Semantic Scholar (会议/期刊)...")
    papers =[]
    # 选取一个宽泛的关键词作为示例
    url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    params = {
        'query': 'deepfake detection OR text-to-video',
        'fields': 'title,url,abstract,venue,publicationDate',
        'publicationTypes': 'JournalArticle,Conference',
        'year': f'{datetime.date.today().year}',
        'limit': 5
    }
    try:
        response = requests.get(url, params=params).json()
        if 'data' in response:
            for item in response['data']:
                if item.get('abstract'):
                    papers.append({
                        "title": item['title'],
                        "url": item['url'] if item.get('url') else "无链接",
                        "abstract": item['abstract'],
                        "source": f"Semantic Scholar - {item.get('venue', 'Unknown Venue')}"
                    })
    except Exception as e:
        print(f"Semantic Scholar 检索出错: {e}")
    return papers

def analyze_with_llm(paper):
    """调用大模型提炼论文信息"""
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    prompt = f"""
    你是一个顶级的多媒体安全与AIGC研究助手。请阅读以下英文论文的标题和摘要：
    Title: {paper['title']}
    Abstract: {paper['abstract']}
    
    请按以下严格格式输出（全中文）：
    📌 **标题：** [中文标题翻译] ({paper['title']})
    🔗 **链接：** {paper['url']}
    📚 **来源：** {paper['source']}
    📝 **摘要：**[用100字左右的中文简明扼要地概括摘要的核心内容]
    ✨ **亮点：**[根据摘要分析这篇论文的创新点、采用了什么方法、或者解决了什么具体问题，用一到两句话直击痛点]
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"大模型处理失败: {e}"

def send_email(content):
    """发送 HTML 格式的精美邮件"""
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = f"📚 每日学术速递 (I2V/T2V & 伪造防御) - {datetime.date.today()}"
    
    # 简单的 HTML 排版
    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2 style="color: #2C3E50;">🚀 您的每日专属论文列表已生成！</h2>
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
        print("✅ 邮件发送成功！")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")

if __name__ == "__main__":
    # 1. 收集论文
    all_papers = fetch_arxiv_papers() # + fetch_semantic_scholar() 可选合并，目前ArXiv最新
    
    # 去重
    seen_titles = set()
    unique_papers =[]
    for p in all_papers:
        if p['title'] not in seen_titles:
            seen_titles.add(p['title'])
            unique_papers.append(p)
            
    # 我们每天只选取最新的，避免冗长（这里限制前 8 篇）
    unique_papers = unique_papers[:8]
    
    if not unique_papers:
        print("今日无相关论文更新。")
        send_email("今日没有发现符合您方向的新论文发布，请好好休息！")
        exit()
        
    # 2. 大模型分析汇总
    final_report = ""
    for idx, paper in enumerate(unique_papers):
        print(f"正在处理第 {idx+1}/{len(unique_papers)} 篇...")
        analysis = analyze_with_llm(paper)
        final_report += analysis + "\n\n" + "—" * 40 + "\n\n"
        
    # 3. 发送邮件
    send_email(final_report)