from pathlib import Path
import json,re,html,os
from PIL import Image as PILImage
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.platypus import SimpleDocTemplate,BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether,Preformatted
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY,TA_LEFT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
P=Path(os.environ.get('REPORT_ROOT',Path(__file__).resolve().parents[1])).resolve();D=P/'deliverables';D.mkdir(exist_ok=True);blocks=json.loads((P/'report_blocks.json').read_text())
REPORT_BASENAME=os.environ.get('REPORT_BASENAME','Andreev_Lab4_Report')
REPORT_HEADING=os.environ.get('REPORT_HEADING','Распознавание и анализ объектов\nна изображениях и видео').replace('\\n','\n')
REPORT_SUBJECT=os.environ.get('REPORT_SUBJECT','Эксперименты на открытых изображениях и видео бактерий')
for family,prefix in [('Serif','DejaVuSerif'),('Sans','DejaVuSans'),('Mono','DejaVuSansMono')]:
 base=Path('/usr/share/fonts/truetype/dejavu')
 pdfmetrics.registerFont(TTFont(family,str(base/(prefix+'.ttf'))));pdfmetrics.registerFont(TTFont(family+'Bold',str(base/(prefix+'-Bold.ttf'))));pdfmetrics.registerFontFamily(family,normal=family,bold=family+'Bold',italic=family,boldItalic=family+'Bold')
styles={
 'body':ParagraphStyle('Body',fontName='Serif',fontSize=12.5,leading=18.5,alignment=TA_JUSTIFY,firstLineIndent=24,spaceAfter=8),
 'h1':ParagraphStyle('H1',fontName='SerifBold',fontSize=15,leading=20,spaceAfter=14,keepWithNext=True),
 'h2':ParagraphStyle('H2',fontName='SerifBold',fontSize=12.5,leading=17,spaceBefore=12,spaceAfter=8,keepWithNext=True),
 'cell':ParagraphStyle('Cell',fontName='Serif',fontSize=9.4,leading=12,spaceAfter=0,splitLongWords=True),
 'head':ParagraphStyle('Head',fontName='SerifBold',fontSize=9.4,leading=12,spaceAfter=0),
 'caption':ParagraphStyle('Caption',fontName='Serif',fontSize=10.5,leading=14,alignment=TA_CENTER,spaceBefore=6,spaceAfter=14),
 'title':ParagraphStyle('Title',fontName='SerifBold',fontSize=18,leading=26,alignment=TA_CENTER,spaceAfter=16),
 'center':ParagraphStyle('Center',fontName='Serif',fontSize=12,leading=17,alignment=TA_CENTER,spaceAfter=8),
 'small':ParagraphStyle('Small',fontName='Serif',fontSize=10,leading=14,alignment=TA_LEFT),
 'code':ParagraphStyle('Code',fontName='Mono',fontSize=8,leading=11,spaceBefore=4,spaceAfter=10,backColor=colors.HexColor('#f3f4f6'),leftIndent=5,rightIndent=5),
}
def esc(t):return html.escape(t).replace('\n','<br/>')
class Report(BaseDocTemplate):
 def __init__(self,path):
  super().__init__(str(path),pagesize=A4,leftMargin=72,rightMargin=45,topMargin=55,bottomMargin=48,title='Лабораторная работа № 4 — Андреев А. К.',author='Андреев А. К.',allowSplitting=1)
  self.addPageTemplates(PageTemplate(id='normal',frames=[Frame(self.leftMargin,self.bottomMargin,self.width,self.height,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.page))
 def page(self,c,d):
  if d.page>1:
   c.setFont('Serif',10);c.drawCentredString(A4[0]/2,27,str(d.page))
 def afterFlowable(self,flowable):
  if isinstance(flowable,Paragraph) and flowable.style.name == 'H1' and flowable.getPlainText() != 'Содержание':
   level=0 if flowable.style.name=='H1' else 1;text=flowable.getPlainText();self.notify('TOCEntry',(level,text,self.page))

doc=Document();sec=doc.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.top_margin=Cm(2);sec.bottom_margin=Cm(2);sec.left_margin=Cm(3);sec.right_margin=Cm(1.5)
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(14);normal.paragraph_format.line_spacing=1.5;normal.paragraph_format.first_line_indent=Cm(1.25);normal.paragraph_format.space_after=Pt(4)
for name,size in [('Heading 1',16),('Heading 2',14)]:
 st=doc.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0);st.paragraph_format.first_line_indent=Cm(0);st.paragraph_format.keep_with_next=True
sec.different_first_page_header_footer=True;fp=sec.footer.paragraphs[0];fp.alignment=WD_ALIGN_PARAGRAPH.CENTER;field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');fp._p.append(field)
settings=doc.settings.element;upd=OxmlElement('w:updateFields');upd.set(qn('w:val'),'true');settings.append(upd)
story=[]
def titleline(text,size=12,bold=False,space=8):
 pa=doc.add_paragraph();pa.alignment=WD_ALIGN_PARAGRAPH.CENTER;pa.paragraph_format.first_line_indent=Cm(0);pa.paragraph_format.line_spacing=1;pa.paragraph_format.space_after=Pt(space);rr=pa.add_run(text);rr.bold=bold;rr.font.size=Pt(size)
 story.append(Paragraph(esc(text),styles['title'] if size>=17 else styles['center']))
titleline('МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ\nРОССИЙСКОЙ ФЕДЕРАЦИИ',11)
titleline('Казанский национальный исследовательский\nтехнический университет им. А. Н. Туполева — КАИ',12)
titleline('Институт компьютерных технологий и защиты информации\nКафедра прикладной математики и информатики',11)
for _ in range(2):doc.add_paragraph('')
story.append(Spacer(1,48))
titleline('ЛАБОРАТОРНАЯ РАБОТА № 4',18,True)
titleline('по дисциплине «Компьютерное зрение»',14)
titleline(REPORT_HEADING,16,True)
titleline('Отчёт о выполненных экспериментах',12)
for _ in range(2):doc.add_paragraph('')
story.append(Spacer(1,50))
titleline('Выполнил: студент группы 4299\nАндреев А. К.',14)
titleline('Проверила: к. т. н., доцент\nБикмуллина И. И.',14)
story.append(Spacer(1,50));titleline('Казань, 2026',14)
doc.add_page_break();story.append(PageBreak())
doc.add_heading('Содержание',level=1);story.append(Paragraph('Содержание',styles['h1']))
# Word field is editable and updates when opened in Word; full heading list remains visible.
for v in blocks:
 if v['kind']=='h' and v['level']==1:
  pa=doc.add_paragraph(v['text']);pa.paragraph_format.first_line_indent=Cm(0);pa.paragraph_format.line_spacing=1;pa.paragraph_format.space_after=Pt(5);pa.runs[0].font.size=Pt(12)
toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc1',fontName='Serif',fontSize=11,leading=15,spaceBefore=5,leftIndent=0),ParagraphStyle('toc2',fontName='Serif',fontSize=10,leading=13,leftIndent=15)];story.append(toc)
fig_no=0;table_no=0
for v in blocks:
 kind=v['kind']
 if kind=='h':
  if v['level']==1:doc.add_page_break();story.append(PageBreak())
  doc.add_heading(v['text'],level=v['level']);story.append(Paragraph(esc(v['text']),styles['h'+str(v['level'])]))
 elif kind=='p':
  pa=doc.add_paragraph(v['text']);pa.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY;story.append(Paragraph(esc(v['text']),styles['body']))
 elif kind=='code':
  pa=doc.add_paragraph();pa.paragraph_format.first_line_indent=Cm(0);pa.paragraph_format.line_spacing=1;pa.paragraph_format.space_after=Pt(8);run=pa.add_run(v['text']);run.font.name='DejaVu Sans Mono';run.font.size=Pt(8)
  # Preserve source line breaks and wrap long commands at spaces in the PDF.
  import textwrap
  lines=[]
  for line in v['text'].splitlines():lines.extend(textwrap.wrap(line,92,replace_whitespace=False,drop_whitespace=False) or [''])
  story.append(Preformatted('\n'.join(lines),styles['code']))
 elif kind=='table':
  table_no+=1;label=f'Таблица {table_no}';pa=doc.add_paragraph(label);pa.paragraph_format.first_line_indent=Cm(0);pa.paragraph_format.keep_with_next=True;story.append(Paragraph(label,styles['small']))
  t=doc.add_table(rows=1,cols=len(v['headers']));t.style='Table Grid';t.autofit=False;ratios=v['widths'] or [1/len(v['headers'])]*len(v['headers'])
  for column,ratio in zip(t.columns,ratios):column.width=Cm(16.5*ratio)
  for i,(cell,text) in enumerate(zip(t.rows[0].cells,v['headers'])):cell.text=text
  for row in v['rows']:
   for cell,text in zip(t.add_row().cells,row):cell.text=text
  for ri,row in enumerate(t.rows):
   # Repeat table header on new pages.
   if ri==0:
    rep=OxmlElement('w:tblHeader');row._tr.get_or_add_trPr().append(rep)
   for ci,cell in enumerate(row.cells):
    cell.width=Cm(16.5*ratios[ci])
    for pa in cell.paragraphs:
     pa.paragraph_format.first_line_indent=Cm(0);pa.paragraph_format.line_spacing=1;pa.paragraph_format.space_after=Pt(3);pa.paragraph_format.space_before=Pt(3)
     for rr in pa.runs:rr.font.name='Times New Roman';rr.font.size=Pt(10);rr.bold=ri==0
  data=[[Paragraph(esc(x),styles['head']) for x in v['headers']]]+[[Paragraph(esc(x),styles['cell']) for x in row] for row in v['rows']]
  tpdf=Table(data,colWidths=[478*x for x in ratios],repeatRows=1,hAlign='LEFT');tpdf.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.HexColor('#9aa0a6')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef1f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.extend([tpdf,Spacer(1,12)])
 elif kind=='figure':
  fig_no+=1;im=PILImage.open(v['path']);w,h=im.size;scale=min(478/w,340/h);caption=f'Рисунок {fig_no} — '+v['caption']
  pa=doc.add_paragraph();pa.alignment=WD_ALIGN_PARAGRAPH.CENTER;pa.paragraph_format.first_line_indent=Cm(0);pa.paragraph_format.keep_with_next=True;pa.add_run().add_picture(v['path'],width=Cm(min(16.3,12*h/w if False else 16.3)),height=None) if h/w<1.2 else pa.add_run().add_picture(v['path'],height=Cm(12))
  cp=doc.add_paragraph(caption);cp.alignment=WD_ALIGN_PARAGRAPH.CENTER;cp.paragraph_format.first_line_indent=Cm(0);cp.paragraph_format.line_spacing=1;cp.paragraph_format.space_after=Pt(10)
  for rr in cp.runs:rr.font.size=Pt(11)
  story.append(KeepTogether([Image(v['path'],width=w*scale,height=h*scale),Paragraph(esc(caption),styles['caption'])]))
doc.core_properties.author='Андреев А. К.';doc.core_properties.title='Лабораторная работа № 4. Компьютерное зрение';doc.core_properties.subject=REPORT_SUBJECT;doc.save(D/f'{REPORT_BASENAME}.docx')
Report(D/f'{REPORT_BASENAME}.pdf').multiBuild(story)
print('Saved DOCX and PDF',fig_no,'figures',table_no,'tables')
