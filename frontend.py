import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64

model = load_model('xceptionforleaf.h5')
class_dict = np.load("xceptionforleaf.npy")


def predict(image):
    IMG_SIZE = (1, 299, 299, 3)

    img = image.resize(IMG_SIZE[1:-1])
    img_arr = np.array(img)
    img_arr = img_arr.reshape(IMG_SIZE)

    pred_proba = model.predict(img_arr)
    pred = np.argmax(pred_proba)
    print(pred)
    return pred



# Streamlit App
st.title("Medicinal Leaf  Classification ")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    st.write("")
    st.write("Classifying...")
    pred = predict(img)
    predicted_class = class_dict[pred]
    # Map index to class name
   
    uses_of_the_plant = {"Aloevera":" → It is mainly used as a remedy for constipation, colic, skin diseases, worm infestations and infections in traditional Indian medicine.",
	'Amla':" → Amla is a natural energiser that enhances energy and stamina by relieving fatigue, stress, eliminating toxins from the body. It can be consumed as whole fruit juice, or Amla powder can be mixed with water, smoothies, tea or soups ",
        "Amruthaballi":" → Amrutha Balli is widely used in Ayurveda in different preparations. From juices to powder, the list is endless. The active principles found in the powder are supposed to be antioxidant, antimicrobial, antitoxic, antidiabetic, anticancer, antistress, anti-inflammatory, anti-allergic etc.",
        "Arali":" → Despite the danger, oleander seeds and leaves are used to make medicine. Oleander is used for heart conditions, asthma, epilepsy, cancer, painful menstrual periods, leprosy, malaria, ringworm, indigestion ",
        "Astma_weed":" → Asthma weed has been used for the remediation of respiratory diseases, some female diseases, and others such as dysentery, jaundice, gonorrhea, pimples, tumors, digestive problems, and childhood infections.",
        "Badipala":" → Badipala is usually a much-branched somewhat climbing shrub, rarely a small tree. Leaves are ovate-oblong to elliptic, 1-5 cm long, 0.7-3 cm wide, produced on short lateral branchlets, looking like leaflets of a compound leaf.",
        "Balloon_Vine":" → Balloon Vine helps to treat arthritis and it greatly helps to reduce the Inflammation and Joint Pain. The extract of Balloon Vine is a good herbal treatment for Cancer. ",
        "Bamboo":" → Fung, Fargesia spathacea Franch., Pleioblastus amarus (Keng) Keng f. and Sasa palmata (Burb.) E.G.Camus are the well-known species among them. These bamboo species, especially their leaves have been a staple of medicine in China, Japan and Southeast Asia since ancient times.",
        "Beans":" → Fung, Fargesia spathacea Franch., Pleioblastus amarus (Keng) Keng f. and Sasa palmata (Burb.) E.G.Camus are the well-known species among them. These bamboo species, especially their leaves have been a staple of medicine in China, Japan and Southeast Asia since ancient times. ",
        "Betel":" → Betel leaves may help in relieving headaches, fighting against cancer, healing wounds, may reduce gastric ulcers, diabetes, and allergies.",
        "Bhrami":" → Brahmi may be useful in managing anxiety due to its anxiolytic (anti-anxiety) property. It may reduce the symptoms of anxiety and mental fatigue while increasing memory span. Brahmi may also prevent neuroinflammation.",
        "Bringaraja":" → Bhringraj helps to control premature greying of hair. It has an ability to rejuvenate hair due to its Rasayana property. Bhringraj helps in quick healing of the wound, decreases swelling and brings back the normal texture of the skin..",
        "Caricature":" → It has anti-inflammatory, analgesic, anti-diabetic, oxytocic, and nephroprotective properties. . The leaves are infused as remedy for constipation..",
        "Castor":" → Castor bean has been mainly recommended as anti-inflammatory, anthelmintic, anti-bacterial, laxative, abortifacient, for wounds, ulcers, and many other indications. ",
        "Catharanthus":" → C. roseus can be considered as a rich source of alkaloids and phenolics, which possess diverse biological properties including anticancer, antidiabetic, antioxidant, antimicrobial and antihypertensive activities.. ",
        "Chakte":" → Effectively gives relief on skin rashes, redness on sensitive skin and soothes the skin and affected area.Gives fairer, smoother and radiant skin. ",
	'Chilly':" → Chili plays important role to increase immunity, anti-ulcer, analgesic, anti-inflammatory, and anti-hemorrhoid agent..",
        "Citron lime ":" → Its slightly bitter taste is a wonderful palate cleanser, and is known as 'pitta-ahara' in Ayurvedic texts, the remedy for 'pitha dosha' characterised by conditions of nausea, acidity and indigestion.",
        "Coffee":" →the presence of antioxidants helps in improving brain health and increases alertness, and also helps in managing weight. According to Ayurveda, Coffee has Rajas properties, which helps in boosting the energy levels. ",
        "Common rue":" → it is used for digestion problems including loss of appetite, upset stomach, and diarrhea. It is also used for heart and circulation problems.",
        "Coriender":" → it might help with digestive problems, abdominal discomforts, and loss of appetite. The leaves of dhania may be used as an appetiser.",
        "Curry":" → it may have a blood pressure-lowering effect.It may have antibacterial activity.It may have antiviral activity.It may have antifungal activity.",
        "Doddpathre":" → Doddapatre / Ajwain is a famous medicinal plant, the leaves of which are used to treat cough, cold, sore throat, nasal congestion and fever. Doddapatre contains powerful expectorants that help to eliminate mucus and phlegm from your respiratory tracts. It is used in skin treatment. It possesses anti-inflammatory compounds that can quickly reduce redness and swelling while also eliminating itchiness and irritation. ",
        "Drumstick":" → It may have an anti-oxidant property.It may be an anti-diabetic (reduces blood glucose levels)It may have anti-cancer (prevents the growth of cancer cells) potential. ",
        'Ekka':" → Ekka is a perennial herb, climbing or trailing up to 3 m, stem bristly-hairy. Tendrils are simple, thread-like. Leaves are arrow shaped, hastate, ...",
        "Eucalyptus":" → Eucalyptus oil is useful if you have cough related problems like bronchitis. In Ayurveda, this disease is known as Kasroga. A massage with Eucalyptus oil reduces excess mucus accumulation and reduces inflammation because of its Kapha balancing and Ushna (hot) properties.",
        "Ganigale":" →  Ganigale, is a semi-succulent perennial plant in the family Lamiaceae with a pungent oregano-like flavor and odor",
        "Ganike":" → ganika is a species of poppy found in Mexico and now widely naturalized in many parts of the world. An extremely hardy pioneer plant, it is tolerant of drought and poor soil, often being the only cover on new road cuttings or verges. It has bright yellow latex. ",
        "Gasagase":" → Gasagase, the mountain knotgrass, is a woody, prostrate or succulent, perennial herb in the family Amaranthaceae, native to Asia, Africa. It has been included as occurring in Australia by the US government, but it is not recognised as occurring in Australia by any Australian state herbarium.",
        "Ginger":" →  Promotes healthy hair growth.Supports healthy skin and a clear complexion.Removes excess heat from the body.Supports proper function of the kidneys.. The leaves and young fruits are used as a vegetable, the dried leaves are used for tea and as a soup thickener, and the seeds are edible.",
        "Globe Amarnath":" → Regardless of genus or species, night-blooming cereus flowers are almost always white or very pale shades of other colors, often large, and frequently fragrant. Most of the flowers open after nightfall, and by dawn, most are in the process of wilting.",
        "Guava":" → Guava helps to manage colic pain when taken with food. Colic pain begins in the abdomen and often radiates to the groin. As per Ayurveda, Vata may produce colic pain in the colon causing difficulty in passing stool.",
        "Henna":" → henna has been used for severe diarrhea caused by a parasite (amoebic dysentery), cancer, enlarged spleen, headache, jaundice, and skin conditions. These days, people take henna for stomach and intestinal ulcers..",
        "Hibiscus":" →Promotes healthy hair growth.Supports healthy skin and a clear complexion.Removes excess heat from the body.Supports proper function of the kidney.",
        'Honge':" → It is a very good brain stimulant. It removes chest, nose and head congestion. It relieves cough and cold. It relieves dyspepsia impairment of digestion",
        "Insulin":" → Tephrosia purpurea is a species of flowering plant in the family Fabaceae, that has a pantropical distribution. It is a common wasteland weed. In many parts it is under cultivation as green manure crop. It is found throughout India and Sri Lanka in poor soils. Common names include: Bengali",
        "Jackfruit":" → It is a rich source of antioxidants, vitamins, and minerals, and is said to have purifying and cooling properties that can help regulate body temperature and improve digestion. ",
        "Jasmine":" → Jasmine has been used for liver disease (hepatitis), liver pain due to cirrhosis, and abdominal pain due to severe diarrhea (dysentery). It is also used to cause relaxation (as a sedative), to heighten sexual desire (as an aphrodisiac), and in cancer treatment .",
        "Kambajala":" →  Kambajala are fast growers, with vestita and quadrifolia ranging between 4 inches and one foot of maximum height, depending on water depth ...",
        "Kasambruga":" → It is known by many common names including Shona cabbage, African cabbage, spiderwisp, cat's whiskers, chinsaga and stinkweed. It is an annual wildflower native to Africa but has become widespread in many tropical and sub-tropical parts of the world.",
        "Kohlrabi":" → This site makes an attempt to gather and share common names of the plants found in India. The common names are just as important as the scientific names.",
        "Lantana":" → Native to South America, stinking passionflower is a climbing vine with an unpleasant smell and flowers that resemble those of the passionfruit vine. Stinking passionflower can invade forest edges, coastal vegetation, roadsides and disturbed areas. It is widespread in northern Queensland.",
        "Lemon":" → he intake of Lemon with salt is a common remedy to help manage nausea as it helps to promote digestion. Lemon essential oil mixed with some other carrier oil like olive oil helps to reduce stress.",
        "Lemon Grass":" → Drinking Lemongrass tea (kadha) twice a day helps in weight loss as it removes toxins from the body and improves metabolism. Applying Lemongrass oil to the skin in combination with some carrier oil like olive oil or coconut oil helps to get relief from pain and swelling due to its anti-inflammatory property.",
        "Malabar_Nut":" → Malabar nut is  an ornamental plant in the genus Senna. It is used in herbalism. It grows natively in upper Egypt, especially in the Nubian region, and near Khartoum, where it is cultivated commercially. It is also grown elsewhere, notably in India and Somalia. ",
        'Malabar_Spinach':" → Unlike a tree, vines can't support themselves, so a trellis provides this support. Also, trellises keep vines off the ground and therefore minimize disease. They help spread out the canopy for sun exposure, pruning and canopy management.",
        "Mango":" →Mango possesses antidiabetic, anti-oxidant, anti-viral, cardiotonic, hypotensive, anti-inflammatory properties. ",
        "MariGold":" → Marigold is considered cooling in nature and thus balancing for Pitta as well as Kapha Doshas.",
        "Mint":" → Mint is known as pudina in Sanskrit. It's a popular cooling herb with a sweet taste and a pungent aftertaste.",
        "Neem":" → Azadirachta indica, commonly known as neem, nimtree or Indian lilac, is a tree in the mahogany family Meliaceae. It is one of two species in the genus Azadirachta, and is native to the Indian subcontinent. It is typically grown in tropical and semi-tropical regions. Neem trees also grow on islands in southern Iran.",
        "Nelavembu":" → Nilavembu helps to manage blood sugar levels and is useful for people suffering from diabetes. It also helps fight cancer and detoxifies the liver. Its rich source of antimicrobial and antiviral properties help manage all kinds of fever including dengue, typhoid, influenza, malaria and chikungunya.", 
        "Nerale":" → Jamun can help in recovering from diarrhoea and indigestion and is widely used in Ayurveda and Unani systems of medicine.",
         "Nooni":"Some uses of the fruit include the treatment of inflammation, abscesses, angina, diabetes, ranula, abdominal fibromas, and scorpionfish stings.",
         "Onion":" → In traditional medicine, onion has been used for a large variety of ailments such as headache, fever, toothache, cough, sore throat, flu, baldness, epilepsy, rash, jaundice, constipation, flatulence, intestinal worms, low sexual power, rheumatism, body pain and muscle cramps, high blood pressure, and diabetes.",
        "padri":" →Traditionally, it is mainly used as analgesic, liver stimulant, astringent, wound healing and anti-dyspeptic. Steeped in water they impart their fragrance.",
        "spinach":" → As a medicine, spinach is used to treat stomach and intestinal (gastrointestinal, GI) complaints and fatigue. It is also used as a blood-builder and an appetite stimulant. Some people use it for promoting growth in children and recovery from illness.",
        "papaya":" → Papaya has many benefits, including protection against heart disease, reduced inflammation, aid in digestion, and boosting your immune system. There are also benefits to eating papaya seeds.",
        "parijatha":" → Treats various types of fever.Treat arthritic knee pain and sciatica.Cures dry cough.Anti-allergic, antiviral, and antibacterial properties.Immunity booster.Diabetes Control.Nourishment of hairs..",
        "Pea":" → PEA is used for different types of pain, fibromyalgia, osteoarthritis, multiple sclerosis (MS), carpal tunnel syndrome, autism, and many other conditions.",
        "Black Pepper":" → Black pepper helps in good digestion and when it is consumed raw, hydrochloric acid is released by the stomach and helps in breaking down the particles.",
        "Pomegranate":" → Pomegranate can be used in the prevention and treatment of several types of cancer, cardiovascular disease, osteoarthritis, rheumatoid arthritis, and other diseases. In addition, it improves wound healing and is beneficial to the reproductive system.",
        "Pumpkin":" → Pumpkin provides calcium, potassium, and magnesium, which can help keep your heartbeat regular and your blood pressure low. The fiber in pumpkin can also play a part in lowering blood pressure as well as cholesterol. In addition, the fiber in pumpkin makes you feel full promoting weight loss.",
        "Raddish":" →The root is used as food and also as medicine. Radish is used for stomach and intestinal disorders, bile duct problems, loss of appetite, pain and swelling (inflammation) of the mouth and throat, tendency towards infections, inflammation or excessive mucus of the respiratory tract, bronchitis, fever, colds, and cough.",
        "Rose":" →  Rose flowers are Anti-depressant, anti-spasmodic, aphrodisiac, astringent, increase bile production, cleansing, anti- bacterial and antiseptic. Rose hips tea is also used in the treatment of diarrhoea. Rose petals are mildly sedative, antiseptic, anti- inflammatory, and anti- parasitic.",
        "Sampinga":" →its flowers and stem bark are useful in diabetes, quick wound healing, cardiac disorders, gout, dysuria and more.",
        "Sapota":" → Sapota is rich in vitamin C and antioxidants that help build your immunity. Polyphenol present in sapota may combat detrimental toxins and lowers the risk of diseases. It also has antibacterial and anti-viral properties that act as safeguards the system from harmful microbes.",
        "Seethashoka":" → manage female disorders like dysmenorrhea and menorrhagia due to its Vata balancing property. It also helps to control bleeding in piles due to its Sita (cold) property. Ashoka powder is also an effective remedy for managing worm infestation due to its Krimighna (anti-worm) property.",
        "seethapala ":" →Custard apple is high in fibers, which helps digestion, prevents constipation, and detoxifies our body. Sitafal contains many antioxidants (flavonoids, phenolic compounds, kaurenoic acid, and vitamin C) that fight free radicals associated with chronic diseases, cancer, and heart disease.",
        "spinach ":" → As a medicine, spinach is used to treat stomach and intestinal (gastrointestinal, GI) complaints and fatigue. It is also used as a blood-builder and an appetite stimulant. Some people use it for promoting growth in children and recovery from illness.",
        "Tamarind ":" → Tamarind has played an important role in traditional medicine. In beverage form, it was commonly used to treat diarrhea, constipation, fever, and malaria. The bark and leaves were also used to promote wound healing (1). Modern researchers are now studying this plant for potential medicinal uses.",
        "Taro ":" →Taro corms contain valuable bioactive molecules effective against cancer and cancer-related risk factors, such as carcinogens and biological agents, several pathophysiological conditions, including oxidative stress and inflammation, while controlling metabolic dysfunctions and boosting the immunological response.",
        "Tecoma ":" →Tecoma stans is a herbal medicine used for treatment of diabetes, digestive problems, control of yeast infections, as powerful diuretic, vermifuge and tonic.",
        "Thumbe ":" → This plant has been used for centuries in Chinese medicine to treat various ailments such as fever, cough, sore throat, and colds. The leaves of the plant contain saponins, which are believed to have anti-inflammatory properties. This plant contains saponin compounds, which are known to have anti-inflammatory effects.",
        "Tomato ":" →  Tomato is used for preventing cancer of the breast, bladder, cervix, colon and rectum, stomach, lung, ovaries, pancreas, and prostate. It is also used to prevent diabetes, diseases of the heart and blood vessels (cardiovascular disease), cataracts, and asthma.",
        "Tulsi ":" →  Tulsi is used to treat insect bites. Tulsi is also used to treat heart disease and fever. Tulsi is also used to treat respiratory problems. Tulsi is used to cure fever, common cold and sore throat, headaches and kidney stones.",
        "Turmeric ":" →  In Ayurvedic practices, turmeric is thought to have many medicinal properties including strengthening the overall energy of the body, relieving gas, dispelling worms, improving digestion, regulating menstruation, dissolving gallstones, and relieving arthritis.",
        "Ashoka ":" →   Ashoka helps to manage various gynecological and menstrual problems in women such as heavy, irregular and painful periods. It can be taken in the form of churna/powder or capsule twice a day after meals to get relief from abdominal pain and spasms.",
        "camphor ":" →   Potential uses of Camphor include antiseptic, antipruritic, analgesic (topical), anti-inflammatory, expectorant, cough suppressant, nasal decongestant, contraceptive,, anti-infective, anticancer, and antispasmodic.",
        "Kama Kasturi ":" →   Kama Kasturi seeds fulfil  the RDI of calcium, magnesium, and iron. Magnesium supports healthy bones and muscular function, and iron aids in increasing the formation of red blood cells. Therefore, vegans can also eat these seeds to compensate for their iron and calcium deficiencies.",
        "Kepala ":" → It is also used to prevent scurvy and stimulate digestion . S. acmella is among the most common Amazonian medicinal plants."
        
         }
    st.write(f"Prediction: {predicted_class}")

    if predicted_class in  uses_of_the_plant:
        st.write(f"Use:{uses_of_the_plant[predicted_class]}")
