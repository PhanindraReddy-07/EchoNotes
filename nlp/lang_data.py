"""
Language Identification Training Data
======================================
Training sentences for the EchoNotes Language ID Classifier.

Labels:
  en      - English
  hi      - Hindi (Devanagari script)
  te      - Telugu script
  hi_en   - Hindi-English code-mix (romanized Hindi + English)
  te_en   - Telugu-English code-mix (romanized Telugu + English)

HOW TO ADD YOUR OWN DATA:
  Copy actual sentences from your audio transcripts and paste them
  into the appropriate list below. Even 20-30 extra sentences per
  class will improve accuracy significantly.
  
  Vosk transcripts are unpunctuated and lowercase — add examples
  like that too (see the 'en' list for examples of both styles).
"""

# ── ENGLISH ────────────────────────────────────────────────────────────────
ENGLISH = [
    # Punctuated (lecture notes, documents)
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The process of photosynthesis converts sunlight into chemical energy stored in glucose.",
    "Data structures are used to organize and store data efficiently in computer memory.",
    "Neural networks are computational models inspired by the human brain.",
    "The water cycle describes the continuous movement of water through the environment.",
    "Object-oriented programming uses objects and classes to structure software.",
    "The Internet of Things refers to interconnected devices that communicate over networks.",
    "Recursion is a programming technique where a function calls itself to solve subproblems.",
    "The French Revolution began in 1789 and fundamentally changed European society.",
    "Quantum computing uses quantum mechanical phenomena to perform computations.",
    "Software testing is the process of evaluating a system to identify defects.",
    "Database normalization reduces data redundancy and improves data integrity.",
    "The central nervous system consists of the brain and spinal cord.",
    "Sorting algorithms arrange data in a particular order like ascending or descending.",
    "Cloud computing delivers computing services over the internet on demand.",

    # Vosk-style (unpunctuated, lowercase, no punctuation)
    "machine learning is used in many applications like image recognition and speech processing",
    "the algorithm processes the input data and produces an output based on learned patterns",
    "in this lecture we are going to discuss the basics of operating systems",
    "so today we will cover the topic of linked lists and their operations",
    "the first step is to initialize the variable and then assign a value to it",
    "lets now look at how we can implement this using python code",
    "the time complexity of this algorithm is order n log n",
    "we need to understand the difference between stack and queue data structures",
    "this function takes two parameters and returns the sum of both values",
    "the main advantage of using arrays is that we can access elements in constant time",
    "today we are going to learn about the concept of inheritance in object oriented programming",
    "so the question is how do we handle exceptions in our program",
    "this is an important concept that you will need for your examination",
    "the graph has five nodes and seven edges connecting them",
    "let me explain this with an example to make it clearer",
]

# ── HINDI (Devanagari script) ────────────────────────────────────────────
HINDI = [
    "मशीन लर्निंग एक ऐसी तकनीक है जो कंप्यूटर को डेटा से सीखने में सक्षम बनाती है।",
    "डेटा संरचनाएं कंप्यूटर में डेटा को व्यवस्थित करने के तरीके हैं।",
    "सॉफ्टवेयर इंजीनियरिंग में कोड लिखना और परीक्षण करना शामिल है।",
    "इंटरनेट ने दुनिया को एक वैश्विक गाँव में बदल दिया है।",
    "कृत्रिम बुद्धिमत्ता मानव जैसी सोच को मशीनों में डालने की कोशिश करती है।",
    "डेटाबेस प्रबंधन प्रणाली डेटा को सुरक्षित और व्यवस्थित रखती है।",
    "प्रोग्रामिंग भाषाएं जैसे पायथन और जावा का उपयोग सॉफ्टवेयर बनाने में होता है।",
    "ऑपरेटिंग सिस्टम कंप्यूटर के हार्डवेयर और सॉफ्टवेयर को नियंत्रित करता है।",
    "एल्गोरिदम किसी समस्या को हल करने के लिए चरणों की एक श्रृंखला है।",
    "नेटवर्क सुरक्षा साइबर हमलों से बचाने के लिए महत्वपूर्ण है।",
    "क्लाउड कंप्यूटिंग इंटरनेट के माध्यम से कंप्यूटिंग सेवाएं प्रदान करती है।",
    "रिकर्शन एक ऐसी विधि है जिसमें फंक्शन खुद को बार-बार बुलाता है।",
    "बिग डेटा बहुत बड़ी मात्रा में डेटा को संसाधित करने की तकनीक है।",
    "साइबर सुरक्षा डिजिटल जानकारी को अनधिकृत पहुंच से बचाती है।",
    "आज हम ऑपरेटिंग सिस्टम के बारे में पढ़ेंगे और इसके मुख्य कार्यों को समझेंगे।",
    "इस अध्याय में हम डेटा संरचनाओं के विभिन्न प्रकारों पर चर्चा करेंगे।",
    "पहले हम समस्या को समझेंगे फिर उसका समाधान खोजेंगे।",
    "यह एल्गोरिदम रेखीय समय में काम करता है इसलिए यह कुशल है।",
    "मेमोरी प्रबंधन ऑपरेटिंग सिस्टम का एक महत्वपूर्ण कार्य है।",
    "ग्राफ एक डेटा संरचना है जिसमें नोड्स और एज होते हैं।",
]

# ── TELUGU (Telugu script) ────────────────────────────────────────────────
TELUGU = [
    "మెషిన్ లెర్నింగ్ అనేది కంప్యూటర్ సిస్టమ్‌లు డేటా నుండి నేర్చుకునే సామర్థ్యం.",
    "డేటా స్ట్రక్చర్లు కంప్యూటర్‌లో డేటాను నిర్వహించడానికి ఉపయోగించే పద్ధతులు.",
    "సాఫ్ట్‌వేర్ ఇంజినీరింగ్‌లో కోడ్ రాయడం మరియు పరీక్షించడం ఉంటుంది.",
    "కృత్రిమ మేధస్సు మానవ మెదడు వంటి ఆలోచనను మెషీన్‌లలో అమలు చేయడం.",
    "డేటాబేస్ మేనేజ్‌మెంట్ సిస్టమ్ డేటాను సురక్షితంగా నిల్వ చేస్తుంది.",
    "పైథాన్ మరియు జావా వంటి ప్రోగ్రామింగ్ భాషలు సాఫ్ట్‌వేర్ అభివృద్ధికి ఉపయోగపడతాయి.",
    "ఆపరేటింగ్ సిస్టమ్ కంప్యూటర్ హార్డ్‌వేర్ మరియు సాఫ్ట్‌వేర్‌ను నియంత్రిస్తుంది.",
    "అల్గోరిథమ్ అనేది సమస్యను పరిష్కరించడానికి అనుసరించే దశల శ్రేణి.",
    "నెట్‌వర్క్ భద్రత సైబర్ దాడుల నుండి సిస్టమ్‌లను రక్షిస్తుంది.",
    "క్లౌడ్ కంప్యూటింగ్ ఇంటర్నెట్ ద్వారా కంప్యూటింగ్ సేవలను అందిస్తుంది.",
    "రికర్షన్ అనేది ఒక ఫంక్షన్ తనను తాను పిలుచుకునే పద్ధతి.",
    "లింక్డ్ లిస్ట్ అనేది నోడ్‌ల శ్రేణి ఉండే డేటా స్ట్రక్చర్.",
    "ఈరోజు మనం ఆపరేటింగ్ సిస్టమ్‌ల గురించి నేర్చుకుంటాం.",
    "ఈ అల్గోరిథమ్ లీనియర్ టైమ్‌లో పని చేస్తుంది కాబట్టి ఇది సమర్థవంతమైనది.",
    "ఈ అధ్యాయంలో మనం వివిధ రకాల సార్టింగ్ అల్గోరిథమ్‌లను చర్చిస్తాం.",
    "గ్రాఫ్ అనేది నోడ్స్ మరియు ఎడ్జ్‌లతో కూడిన డేటా స్ట్రక్చర్.",
    "మెమరీ మేనేజ్‌మెంట్ ఆపరేటింగ్ సిస్టమ్‌లో ముఖ్యమైన పని.",
    "పైథాన్‌లో ఫంక్షన్ రాయడం చాలా సులభం మరియు చదవడానికి అర్థమవుతుంది.",
    "డేటా సైన్స్‌లో స్టాటిస్టిక్స్ మరియు ప్రోగ్రామింగ్ రెండూ అవసరం.",
    "నెట్‌వర్కింగ్‌లో TCP/IP ప్రోటోకాల్ చాలా ముఖ్యమైన పాత్ర పోషిస్తుంది.",
]

# ── HINDI-ENGLISH CODE-MIX ───────────────────────────────────────────────
HINDI_ENGLISH = [
    # Romanized Hindi mixed with English (common in speech)
    "aaj hum machine learning ke baare mein padenge aur uske applications dekhenge",
    "yeh algorithm bahut efficient hai kyunki iska time complexity order n log n hai",
    "database mein data store karna aur retrieve karna bahut important hai",
    "is function ko call karte waqt hume parameters pass karne honge",
    "sorting algorithms mein bubble sort aur merge sort ke beech kya difference hai",
    "operating system ka kaam hai hardware aur software ke beech bridge banana",
    "python mein loops likhna bahut easy hai for loop aur while loop dono use kar sakte hain",
    "network security ke liye hume firewall aur encryption use karna chahiye",
    "abhi hum stack data structure ke baare mein discuss karenge",
    "iska main advantage yeh hai ki access time constant hoti hai",
    "jab hum recursion use karte hain toh base case define karna zaruri hai",
    "is problem ko solve karne ke liye hum dynamic programming use karenge",
    "API ke through different services ko connect kar sakte hain",
    "testing ke bina koi bhi software production mein nahi jaana chahiye",
    "cloud computing se hum resources on demand le sakte hain bina hardware khareed kiye",
    # Native Devanagari mixed with English terms
    "यह algorithm बहुत efficient है और इसका use करना आसान है।",
    "Database में data store करते समय proper indexing जरूरी है।",
    "Python में functions लिखना और उन्हें test करना आसान होता है।",
    "इस concept को समझने के लिए एक example देखते हैं।",
    "API endpoints को properly document करना important है।",
]

# ── TELUGU-ENGLISH CODE-MIX ──────────────────────────────────────────────
TELUGU_ENGLISH = [
    # Romanized Telugu mixed with English
    "nenu ippudu machine learning gurinchi chepputhanu mee kosam",
    "ee algorithm time complexity order n log n ga untundi chala efficient",
    "database lo data store chesukovalante proper schema design cheyali",
    "python lo functions rayadam chala easy ga untundi",
    "sorting algorithms lo bubble sort chala slow ga untundi merge sort better",
    "operating system hardware ni manage chestundi software ki interface ista",
    "network lo data transfer chesadam kosam TCP IP protocol use chestam",
    "ikkada mనం recursion gurinchi matladutham base case define cheyali",
    "ee concept artham avvalante oka example chuddam",
    "API ని use chesi different services connect cheyachu",
    "testing cheyakunda software release cheyakoodu",
    "cloud computing lo resources rent cheyachu own cheyakunda",
    "linked list lo nodes oi pointer tho connect avutayi",
    "ee problem ki dynamic programming best solution avutundi",
    "memory management cheyadam operating system important task",
    # Native Telugu mixed with English terms
    "ఈ algorithm చాలా efficient గా పని చేస్తుంది మరియు use చేయడం easy.",
    "Database లో data store చేసేటప్పుడు proper indexing అవసరం.",
    "Python లో functions రాయడం మరియు test చేయడం చాలా సులభం.",
    "ఈ concept అర్థం చేసుకోవడానికి ఒక example చూద్దాం.",
    "API endpoints ని properly document చేయడం important.",
]


# ── COMBINED DATASET ─────────────────────────────────────────────────────

def get_training_data():
    """
    Returns (texts, labels) for training.
    Labels: 'en', 'hi', 'te', 'hi_en', 'te_en'
    """
    texts, labels = [], []

    for s in ENGLISH:
        texts.append(s); labels.append('en')
    for s in HINDI:
        texts.append(s); labels.append('hi')
    for s in TELUGU:
        texts.append(s); labels.append('te')
    for s in HINDI_ENGLISH:
        texts.append(s); labels.append('hi_en')
    for s in TELUGU_ENGLISH:
        texts.append(s); labels.append('te_en')

    return texts, labels


def get_label_info():
    """Returns label descriptions."""
    return {
        'en':    'English',
        'hi':    'Hindi (Devanagari)',
        'te':    'Telugu script',
        'hi_en': 'Hindi-English code-mix',
        'te_en': 'Telugu-English code-mix',
    }
