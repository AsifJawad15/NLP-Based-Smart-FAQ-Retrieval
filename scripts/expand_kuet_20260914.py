"""Apply the manually reviewed 14 September 2026 KUET corpus expansion."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/kuet'
rows = []

def add(question, answer, category, path):
    source = path if path.startswith('https://') else 'https://www.kuet.ac.bd/' + path
    rows.append(dict(id=111 + len(rows), question=question, answer=answer,
                     category=category, source=source, source_type='official_web'))

history = [
('What were the previous names of KUET?', 'Khulna Engineering College (1967-1986), followed by Bangladesh Institute of Technology (BIT), Khulna (1986-2003). The KUET name took effect on 1 September 2003.'),
('Which university was Khulna Engineering College affiliated with?', 'Khulna Engineering College was affiliated with the University of Rajshahi.'),
('How many students joined KUET when teaching first started in 1974?', 'Academic activities began on 3 June 1974 with 120 students, initially 40 in each of three engineering departments.'),
('How many teachers did Khulna Engineering College start with?', 'The college began academic activities in 1974 with nine teachers.'),
('Why was construction of Khulna Engineering College interrupted?', 'Development work was suspended during the Liberation War and resumed after Bangladesh became independent in 1971.'),
('Where was the first temporary canteen at Khulna Engineering College?', 'A temporary canteen was established in the Boiler House of the Mechanical Engineering Department.'),
('When was the connecting road to the KUET campus first constructed?', 'KUET history records that the connecting road from the Khulna-Jessore Highway was first constructed in 1978.'),
('Where did early KUET Mechanical Engineering students take practical classes?', 'First-year Mechanical Engineering practical classes were held at BUET and second-year practical classes at Rajshahi Engineering College while local laboratories were unavailable.'),
('What does Durbar Bangla at KUET commemorate?', 'Durbar Bangla is the campus memorial to the Great Liberation War of Bangladesh.'),
('In which years were the first three KUET convocations held?', 'The first three convocations recorded in KUET history were held in 2006, 2012, and 2018.'),
]
for q,a in history: add(q,a,'university_history','about-kuet/history')

departments = [
('cse','CSE (Computer Science and Engineering)', 'CSE offers undergraduate and postgraduate education in computing, with research in machine learning, natural language processing, security, networks, and computer vision. Students gain practical experience through laboratories and final-year research.'),
('ce','CE (Civil Engineering)', 'Civil Engineering teaches the design, construction, and management of infrastructure such as buildings, bridges, transport systems, and water facilities. The department offers undergraduate and postgraduate education and began undergraduate teaching in 1974.'),
('eee','EEE (Electrical and Electronic Engineering)', 'EEE covers electrical power, electronics, communications, control, embedded systems, and signal processing. It has provided engineering education at KUET since 1974 and supports undergraduate and postgraduate study.'),
('me','ME (Mechanical Engineering)', 'Mechanical Engineering covers machines, manufacturing, thermodynamics, energy, and mechanical dynamics. Its applications include transport, power plants, heating and cooling, and robotics; graduate offerings include M.Sc. Engineering and Ph.D. degrees.'),
('ece','ECE (Electronics and Communication Engineering)', 'ECE provides undergraduate and postgraduate education in electronics and communication systems. Teaching and research include antennas, microwave systems, electronic circuits, digital systems, and signal processing.'),
('iem','IEM (Industrial Engineering and Management)', 'IEM focuses on improving production and service systems through engineering and management. Its department page describes undergraduate Industrial and Production Engineering (IPE) and postgraduate Industrial Engineering and Management programs.'),
('ese','ESE (Energy Science and Engineering)', 'ESE combines mechanical, electrical, and chemical engineering to study energy systems. Its curriculum includes renewable and non-renewable energy, energy conversion, energy efficiency, and sustainable energy solutions.'),
('le','LE (Leather Engineering)', 'Leather Engineering teaches the science and technology of processing hides and skins and producing leather, footwear, and leather products. Its curriculum combines laboratory practice with engineering knowledge.'),
('te','TE (Textile Engineering)', 'Textile Engineering covers fibres, fabric manufacturing, wet processing, apparel, and textile machinery. Its four-year undergraduate program combines chemical and physical principles with textile production technologies.'),
('che','ChE (Chemical Engineering)', 'Chemical Engineering focuses on industrial processes and plants that carry out chemical reactions. The department combines theory with laboratories in areas such as process control, fluid mechanics, heat and mass transfer, and process simulation.'),
('mte','MTE (Mechatronics Engineering)', 'Mechatronics integrates mechanical, electronic, control, computer, and software engineering to create automated machines and intelligent systems. Students study sensors, actuators, drives, PLCs, and control systems alongside practical design.'),
('bme','BME (Biomedical Engineering)', 'Biomedical Engineering combines engineering with medicine and biology. Its work includes medical equipment, physiological measurement, diagnostic systems, biomedical signals, and images.'),
('mse','MSE (Materials Science and Engineering)', 'Materials Science and Engineering studies the links between material structure, processing, properties, and applications. It covers metals, polymers, ceramics, semiconductors, composites, optical and magnetic materials, and nanomaterials.'),
('urp','URP (Urban and Regional Planning)', 'URP studies land use, settlements, transport, public spaces, and regional development. The department emphasizes sustainable, resilient, and equitable communities through planning education and research.'),
('becm','BECM (Building Engineering and Construction Management)', 'BECM combines building design and engineering with construction and project management. It draws on civil engineering, architecture, building services, safety, economics, and environmental considerations throughout a building project.'),
('arch','Architecture', 'Architecture offers a five-year B.Arch program built around design studios, technical and theoretical learning, and a final-year thesis. The department emphasizes climate, culture, sustainability, and the regional built environment.'),
('math','Mathematics', 'Mathematics provides the mathematical foundations needed by KUET engineering students, including modelling and numerical problem solving. It also supports postgraduate education and research in mathematics.'),
('phy','Physics', 'Physics teaches and researches physical science, with postgraduate work in solid-state physics, atmospheric physics, and radiation and health physics. The department also offers M.Sc., M.Phil., and Ph.D. programs.'),
('chem','Chemistry', 'Chemistry provides chemistry teaching for undergraduate engineering students and awards postgraduate degrees. Its education and research concern the composition, structure, reactions, and properties of matter and materials.'),
('hum','Humanities and Business', 'Humanities and Business provides courses for engineering, architecture, and planning students across KUET. Its five main teaching areas are Accounting, Economics, Management, Sociology, and English.'),
]
for key,name,answer in departments:
    add(f'Tell me about the {name} department at KUET.',answer,'department_overviews',key+'/about')

milestones = [
('When was KUET CSE opened and when did academic activities start?', 'CSE began academic activities on 26 September 1999. The department separately lists its establishment date as 14 September 1999.', 'cse/about'),
('When did KUET ECE start undergraduate and postgraduate programs?', 'ECE began undergraduate education in 2001 and postgraduate education in 2011.', 'ece/about'),
('When did KUET IEM start its undergraduate program?', 'IEM began its undergraduate program in 2006 with 30 students.', 'iem/about'),
('When was the KUET ESE department established?', 'Energy Science and Engineering began in May 2007, initially offering M.Sc. Engineering.', 'ese/about'),
('When did KUET ESE start undergraduate admission?', 'ESE introduced its undergraduate program in the 2016-2017 academic session with 30 students.', 'ese/about'),
('When was the KUET Leather Engineering department opened?', 'KUET lists 29 August 2010 as the establishment date of Leather Engineering.', 'le/about'),
('When did the first KUET Textile Engineering batch start classes?', 'The first Textile Engineering batch, from the 2012-2013 session, started classes on 10 March 2013.', 'te/about'),
('When was the KUET Chemical Engineering department established?', 'Chemical Engineering was established on 6 June 2018 under the Faculty of Mechanical Engineering.', 'che/about'),
('When was the KUET Mechatronics Engineering department established?', 'Mechatronics Engineering was founded on 1 July 2018; its first undergraduate batch belonged to the 2018-2019 session.', 'mte/about'),
('When was the KUET Biomedical Engineering department established?', 'Biomedical Engineering was established in May 2007 as a postgraduate department under the Faculty of Electrical and Electronic Engineering.', 'bme/about'),
('When was the KUET Materials Science and Engineering department established?', 'Materials Science and Engineering was established in May 2016 under the Faculty of Electrical and Electronic Engineering.', 'mse/about'),
('When did UGC permit KUET to open the BECM department?', 'UGC permitted the BECM department on 2 September 2013. KUET Academic Council decided on 12 September 2013 to admit 60 undergraduate students for the 2013-2014 session.', 'becm/about'),
]
for q,a,p in milestones: add(q,a,'department_history',p)

cse = [
('Where is the KUET CSE department located?', 'CSE is in Block B of the New Academic Building, KUET, Khulna-9203, Bangladesh.', 'about'),
('What is the KUET CSE department email address?', 'The official departmental contact email is head@cse.kuet.ac.bd (checked 14 September 2026).', 'about'),
('What is the KUET CSE office telephone number?', 'The departmental page lists +8802477733318, extension 350 (checked 14 September 2026).', 'about'),
('Where can I find the KUET CSE faculty list?', 'The official CSE Faculty page lists teaching staff, departmental leadership, faculty on leave, and institutional email addresses.', 'faculty'),
('How many undergraduate seats does KUET CSE list?', 'The CSE department description lists 120 fixed undergraduate places per year. Check the admission circular for the session you are applying to (reviewed 14 September 2026).', 'about'),
('What is CSEA at KUET?', 'CSEA means Computer Science and Engineering Association, the CSE student body that organizes academic, programming, sports, and cultural activities.', 'about'),
('What is SGIPC at KUET?', 'SGIPC means Special Group of Interest in Programming Contest. CSEA uses this group to organize programming contests and prepare students for competitive programming.', 'about'),
('What is HACK at KUET CSE?', 'HACK means Hardware Acceleration Club of KUET. CSEA supports it in organizing hands-on workshops in robotics and embedded systems.', 'about'),
('Can KUET CSE undergraduates participate in research?', 'Yes. The department describes final-year thesis work as an opportunity for undergraduate students to participate in research.', 'about'),
('Which research group is mentioned by KUET CSE?', 'The CSE department description names the Computational Intelligence Research Group and describes research collaboration with other universities.', 'about'),
('Does KUET CSE have a System Development Centre?', 'Yes. The department describes a System Development Centre for developing practical expertise alongside theoretical learning.', 'about'),
('What is the vision of KUET CSE?', 'CSE aims to be a centre of excellence with strong research and teaching that responds to changing computing needs.', 'about'),
('What is the mission of KUET CSE?', 'CSE aims to benefit society through computing education, research, and applications, developing ethical professionals, collaborative researchers, and leaders committed to lifelong learning.', 'about'),
('Which educational objectives does KUET CSE set for graduates?', 'Its four objectives concern professional competence, teamwork and lifelong learning, ethical practice, and diverse careers supported by communication, problem-solving, entrepreneurship, and leadership.', 'ugcurriculum'),
('When should KUET CSE graduates achieve the program educational objectives?', 'The Program Educational Objectives describe achievements expected within three to five years after graduation.', 'ugcurriculum'),
('What program outcomes are expected from KUET CSE graduates?', 'The outcomes cover engineering knowledge, problem analysis, design, investigation, tool use, society, sustainability, ethics, teamwork, communication, project management, and lifelong learning.', 'ugordinance'),
('Which laboratories are listed by KUET CSE?', 'CSE lists ten labs or centres: Computer Language and Computing; Software and Web Engineering; Computer Hardware and Interfacing; Digital Systems and VLSI; Networking and Multimedia; Mobile Games and Apps Development; VDI Multi-Purpose; Mobile Computing; Artificial Intelligence and Robotics; and Natural Language Processing.', 'listoflab'),
('Does KUET CSE have a Natural Language Processing lab?', 'Yes. The official CSE laboratory list includes the Natural Language Processing Laboratory.', 'listoflab'),
('Does KUET CSE have an Artificial Intelligence and Robotics lab?', 'Yes. CSE lists an Artificial Intelligence and Robotics Laboratory.', 'listoflab'),
('Does KUET CSE have a Software and Web Engineering lab?', 'Yes. CSE lists a Software and Web Engineering Laboratory.', 'listoflab'),
('Does KUET CSE have a Digital Systems and VLSI lab?', 'Yes. CSE lists a Digital Systems and VLSI Laboratory.', 'listoflab'),
('Does KUET CSE have a Mobile Games and Apps Development Center?', 'Yes. The Mobile Games and Apps Development Center appears in the official CSE laboratory list.', 'listoflab'),
('What NLP research topics are listed by KUET CSE?', 'Listed NLP topics include text and sentiment mining, neural machine translation, and large language models.', 'researcharea'),
('What machine learning research topics are listed by KUET CSE?', 'CSE lists deep learning, multimodal and federated learning, swarm intelligence, evolutionary computation, pattern recognition, and optimization among its research topics.', 'researcharea'),
('What cybersecurity research topics are listed by KUET CSE?', 'Listed topics include applied cryptography, security protocols, blockchain, data and image encryption, and security in cloud environments.', 'researcharea'),
('What computer vision research is described by KUET CSE?', 'CSE describes image analysis, deep vision, visual modelling, foreground-background modelling, and human activity estimation.', 'researcharea'),
('What cloud and IoT research is described by KUET CSE?', 'CSE lists cloud and fog computing, distributed systems, Internet of Things, and performance analysis of networked systems.', 'researcharea'),
('What healthcare computing research is described by KUET CSE?', 'Listed topics include bioinformatics, biomedical signal processing, medical informatics, and artificial intelligence for health.', 'researcharea'),
]
assert len(cse)==28
for q,a,p in cse: add(q,a,'cse_information','cse/'+p)

faculties = [
('Which departments belong to KUET Civil Engineering faculty?', 'Civil Engineering, Urban and Regional Planning, Building Engineering and Construction Management, and Architecture.', 'dce'),
('Which faculty includes KUET CSE?', 'CSE belongs to the Faculty of Electrical and Electronic Engineering.', 'deee'),
('Which departments belong to KUET EEE faculty?', 'Electrical and Electronic Engineering, Computer Science and Engineering, Electronics and Communication Engineering, Biomedical Engineering, and Materials Science and Engineering.', 'deee'),
('Which departments belong to KUET Mechanical Engineering faculty?', 'Mechanical Engineering, Industrial Engineering and Management, Energy Science and Engineering, Leather Engineering, Textile Engineering, Chemical Engineering, and Mechatronics Engineering.', 'dme'),
('Which departments belong to KUET Science and Humanities faculty?', 'Mathematics, Physics, Chemistry, and Humanities and Business.', 'dsh'),
('Which journal is published by KUET Civil Engineering faculty?', 'The faculty publishes the Journal of Engineering Science (JES).', 'dce'),
('How often is the KUET Journal of Engineering Science published?', 'The Civil Engineering faculty describes JES as published twice yearly, in June and December, since 2010.', 'dce'),
('What are the names of the four KUET faculties?', 'Civil Engineering; Electrical and Electronic Engineering; Mechanical Engineering; and Science and Humanities.', ''),
]
for q,a,p in faculties:add(q,a,'faculties',p)

park='https://ossbhtpa.gov.bd/park-info'
add('Tell me about the IT park at KUET.', 'The Bangladesh Hi-Tech Park Authority lists the KUET facility as the IT Training and Incubation Center. Its directory describes space for training, startups, meetings, and IT businesses within KUET. This is a sourced overview, not a statement of current vacancies or course availability.', 'it_park',park)
add('Where is the KUET IT Training and Incubation Center located?', 'The BHTPA park directory places the IT Training and Incubation Center at Khulna University of Engineering and Technology, Khulna.', 'it_park',park)
add('Does the KUET IT park directory include startup and training space?', 'Yes. The BHTPA directory lists startup space, training rooms, and a meeting room for its KUET centre. Current access and availability require confirmation with the centre.', 'it_park',park)

for q,a,p in [
('Who is the current head of KUET CSE?', 'As checked on 14 September 2026, the CSE About, Faculty, and Head Message pages agree that Professor Dr. Al-Mahmud is the departmental head.', 'cse/faculty'),
('Who is the current head of KUET Civil Engineering?', 'As checked on 14 September 2026, the Civil Engineering faculty list and Head Message page name Professor Dr. Md. Jahir Uddin as departmental head.', 'ce/faculty'),
('Who is the current head of KUET EEE?', 'As checked on 14 September 2026, the EEE faculty list and Head Message page name Professor Dr. Mostafa Zaman Chowdhury as departmental head.', 'eee/faculty'),
('Who is the current head of KUET ECE?', 'As checked on 14 September 2026, the ECE About, Faculty, and Head Message pages name Professor Dr. Monir Hossen as departmental head.', 'ece/faculty'),
('Which office serves as the Chancellor of KUET?', 'The President of the People\'s Republic of Bangladesh serves as Chancellor of KUET.', 'provc'),
]: add(q,a,'university_administration',p)
for name,full,year in [('IICT','Institute of Information and Communication Technology',2010),('IDM','Institute of Disaster Management',2014),('IEPT','Institute of Environment and Power Technology',2016)]:
    add(f'When was KUET {name} opened?',f'The {full} ({name}) opened in {year}, according to KUET history.','institute_history','about-kuet/history')
add('What is the full form of KUET?', 'KUET stands for Khulna University of Engineering and Technology.','university_information','about-kuet/history')

assert len(rows)==90, len(rows)
club_replacements = [
    (114, 'Tell me about KUET Career Club (KCC).', 'KUET Career Club organizes student development activities. KUET reports that it organized BizBattle: Innovation in Tech, a national idea case competition whose January 2025 finale featured ideas on technology, business, environment, and product development.', 'https://www.kuet.ac.bd/news/26'),
    (115, 'Tell me about KUET clubs and student activities.', 'KUET supports student-led activities through the Office of Student Welfare, which coordinates activities, offers advice to student bodies, runs leadership development programs, and liaises with student associations. Ask this FAQ about individual groups such as CSEA, SGIPC, HACK, Career Club, Debating Society, KBEC, IRCC, KUET Theatre, or CADers.', 'https://www.kuet.ac.bd/dsw'),
    (116, 'Tell me about KUET Debating Society (KDS).', 'KUET Debating Society is a student debating organization. Official KUET Fab Lab records document its Bangla and English parliamentary debating activities and inter-university debate competitions. Those records describe past activities, not a current committee or event schedule.', 'https://www2.kuet.ac.bd/fablab/operators-2017/'),
    (117, 'Tell me about KUET Business and Entrepreneurship Club (KBEC).', 'KBEC is the KUET Business and Entrepreneurship Club. Its own website describes a focus on entrepreneurship education, innovation initiatives, and building a business community at KUET.', 'https://www.kbec-official.org/'),
    (118, 'Tell me about IEM Robotics and CAD Club (IRCC) at KUET.', 'IRCC stands for IEM Robotics and CAD Club. KUET Fab Lab records document robotics and programming workshops and an intra-university idea fair organized through IRCC; this is historical activity information.', 'https://www2.kuet.ac.bd/fablab/operators-2018/'),
    (187, 'Tell me about KUET Theatre.', 'KUET Theatre is documented in official KUET Fab Lab records as a student organization involved in cultural programs. The cited 2018 record describes past organizing experience and does not establish its current committee or schedule.', 'https://www2.kuet.ac.bd/fablab/operators-2018/'),
    (191, 'Tell me about CADers at KUET.', 'KUET Fab Lab records document CADers organizing SolidWorks and AutoCAD workshops in the Mechanical Engineering computer lab in 2016. This provides a historical example of its computer-aided design activities.', 'https://www2.kuet.ac.bd/fablab/operators-2017/'),
]
for faq_id,q,a,source in club_replacements:
    rows[faq_id-111].update(question=q,answer=a,category='student_clubs',source=source)
for row in rows:
    if row['question'] in ['What is CSEA at KUET?', 'What is SGIPC at KUET?', 'What is HACK at KUET CSE?']:
        row['category']='student_clubs'
# Avoid a disputed undergraduate degree title in the IEM overview too.
rows[126-111]['answer']='IEM focuses on improving production and service systems through engineering and management. Its interdisciplinary curriculum addresses system design, control, evaluation, and improvement while considering human and environmental needs.'
with (DATA/'faq_dataset.csv').open(encoding='utf-8', newline='') as handle:
    old=list(csv.DictReader(handle))
assert len(old) in (110,200), 'Expected the original or expanded KUET corpus.'
old=old[:110]
old[47]['answer']=old[47]['answer'].replace('11 September 2026','14 September 2026')
# IEM degree naming differs between the central and departmental pages: omit
# that disputed degree title and retain the verified existence of the department.
old[14]['question']='Does KUET have an Industrial Engineering and Management department?'
old[14]['answer']='Yes. KUET has a Department of Industrial Engineering and Management under the Faculty of Mechanical Engineering.'
old[14]['source']='https://www.kuet.ac.bd/dme'
with (DATA/'faq_dataset.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=old[0].keys());writer.writeheader();writer.writerows(old+rows)

review=DATA/'SOURCE_REVIEW.md'
review.write_text(review.read_text(encoding='utf-8').split('\n## Expansion reviewed')[0],encoding='utf-8')
with review.open('a',encoding='utf-8') as handle:
    handle.write('\n## Expansion reviewed 14 September 2026\n\nExactly 200 FAQ rows; IDs 111-200 are manually authored paraphrases and summaries of the sources below. Department overviews are stored summaries, not live generated text. Public source snapshots are under `source_snapshots/2026-09-14/`. Retrieval remains local.\n\n')
    handle.write('Excluded: conflicting EEE dean names; conflicting BME and ESE head names; incomplete Pro-VC template; uncorroborated leadership names; conflicting IEM opening year; ambiguous BME undergraduate starting year; inconsistent IT park area units; placeholder CSE clubs and research links. No VC, named Chancellor, or dean FAQ was added. Four departmental heads were retained only after matching official personnel and head-message pages. All names are explicitly dated.\n\n')
    handle.write('Establishment dates and first academic activity dates are different milestones. CSE dates are explicitly distinguished. FAQ 15 now describes the verified IEM department instead of repeating inconsistent degree naming; FAQ 48 was rechecked on the admission portal. Existing FAQs 1-110 otherwise retain their earlier review date.\n\n')
    handle.write('| FAQ ID | Question / reviewed fact | Official source |\n| --- | --- | --- |\n')
    for row in rows:handle.write(f"| {row['id']} | {row['question']} | {row['source']} |\n")
print('Wrote',len(old+rows),'FAQs')
