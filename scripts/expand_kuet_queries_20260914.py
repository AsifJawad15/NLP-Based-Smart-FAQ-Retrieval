"""Developer-authored query coverage; these are not human evaluation results."""
import csv
from pathlib import Path

data = Path(__file__).resolve().parents[1] / 'data/kuet'
validation = [
('What names did KUET use before becoming a university?',111),
('Which university was Khulna Engineering College affiliated to?',112),
('How many students were in the initial 1974 intake?',113),
('Tell me about KUET career club',114),
('Tell me about KUET clubs',115),
('What is KUET Debating Society?',116),
('Tell me about KBEC at KUET',117),
('What does the IEM Robotics and CAD Club do?',118),
('What does Durbar Bangla represent?',119),
('Which years saw the first three convocations?',120),
('Tell me about CSE dept',121),
('Tell me about civil engineering department',122),
('Tell me about EEE department',123),
('Tell me about mechanical engineering department',124),
('Tell me about ECE department',125),
('Tell me about IEM department',126),
('Tell me about ESE department',127),
('Tell me about leather engineering department',128),
('Tell me about textile engineering department',129),
('Tell me about chemical engineering department',130),
('When was CSE opened?',141),
('When did ECE launch postgraduate study?',142),
('When did IEM introduce undergraduate education?',143),
('In which year was ESE established?',144),
('When did the first textile batch begin classes?',147),
('When was Chemical Engineering founded?',148),
('When was Mechatronics Engineering established?',149),
('Where can I find the CSE department?',153),
('What is the email address of CSE?',154),
('What is CSEA?',158),
('What is SGIPC?',159),
('What is HACK?',160),
('Does CSE have an NLP laboratory?',170),
('What does CSE research in cybersecurity?',177),
('Tell me about the IT park KUET',189),
('Where is the IT incubation centre?',190),
('Who heads CSE?',192),
('Who heads EEE?',194),
('What does KUET stand for?',200),
('What is CADers?',191),
]
validation_negative = [
'Who is the current dean of EEE?',
'Who is the current vice chancellor of KUET?',
'Who currently heads Biomedical Engineering?',
'Tell me about the photography club at KUET.',
'Who is the president of KUET Career Club?',
'What are the membership fees of HACK?',
'What is the current monthly rent at KUET IT park?',
'Give me the current CSE class routine.',
'What is the next SGIPC contest date?',
'What is the password for the CSE lab computers?',
]
smoke = [
('Explain the mathematics department at KUET.',137),
('Tell me about the Physics department.',138),
('Tell me about Chemistry at KUET.',139),
('Tell me about Humanities and Business.',140),
('Tell me about Architecture at KUET.',136),
('Tell me about BECM.',135),
('Tell me about URP.',134),
('Tell me about MSE.',133),
('Tell me about BME.',132),
('Tell me about MTE.',131),
('In what year did ESE begin undergraduate admissions?',145),
('On what date was Leather Engineering opened?',146),
('When was Biomedical Engineering established at KUET?',150),
('When was Materials Science and Engineering established?',151),
('When did UGC give permission to open BECM?',152),
('What telephone number does CSE list?',155),
('Where is the official CSE faculty directory?',156),
('How many undergraduate places are listed for CSE?',157),
('Can CSE undergraduate students do research?',161),
('What research group does CSE mention?',162),
('Is there a System Development Centre in CSE?',163),
('What is the CSE department vision?',164),
('What is the CSE department mission?',165),
('Describe the CSE graduate educational objectives.',166),
('What are the expected CSE program outcomes?',168),
('List CSE laboratories.',169),
('What NLP topics does CSE research?',175),
('What research does CSE do on computer vision?',178),
('Tell me about KUET Theatre',187),
('Which faculty does CSE belong to?',182),
]
smoke_negative = [
'What is the current CSE tuition fee?',
'Who is the current Pro Vice Chancellor?',
'Who is the present dean of Mechanical Engineering?',
'Who currently heads Energy Science and Engineering?',
'What is the swimming pool entry price?',
'Tell me about the KUET astronomy club.',
'What is the next CADers workshop schedule?',
'Does KUET IT park guarantee a job?',
'What is the present hostel room vacancy?',
'Can you predict my CGPA this term?',
]

for filename, original_count, positive, negative in [
    ('validation_queries.csv',50,validation,validation_negative),
    ('smoke_queries.csv',20,smoke,smoke_negative),
]:
    path=data/filename
    with path.open(encoding='utf-8',newline='') as handle:
        old=list(csv.DictReader(handle))[:original_count]
    extra=[dict(query=q,expected_faq_id=i,is_answerable=True) for q,i in positive]
    extra += [dict(query=q,expected_faq_id='',is_answerable=False) for q in negative]
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=old[0].keys());writer.writeheader();writer.writerows(old+extra)
    print(filename,len(old+extra))
