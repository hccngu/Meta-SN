import collections
import json
from collections import defaultdict

import numpy as np
import torch
from torchtext.vocab import Vocab, Vectors

from dataset.utils import tprint



def _get_20newsgroup_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'talk politics mideast': 0,
        'science space': 1,
        'misc forsale': 2,
        'talk politics misc': 3,
        'computer graphics': 4,
        'science  encryption  encrypt secret': 5,
        'computer windows x': 6,
        'computer os ms windows misc': 7,
        'talk politics guns': 8,
        'talk religion misc': 9,
        'rec autos': 10,
        'science med chemistry medical science medicine': 11,
        'computer sys mac hardware': 12,
        'science electronics': 13,
        'rec sport hockey': 14,
        'alt atheism': 15,
        'rec motorcycles': 16,
        'computer system ibm pc hardware': 17,
        'rec sport baseball': 18,
        'soc religion christian': 19,
    }


    train_classes = [0, 3, 8, 9, 2, 15, 19, 17]
    val_classes = [4, 6, 7, 12, 18]
    test_classes = [1, 5, 11, 13, 10, 14, 16]

    return train_classes, val_classes, test_classes, label_dict


def _get_amazon_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'Amazon Instant Video is a subscription video on-demand over-the-top streaming and rental service of Amazon.com': 0,
        'Apps for Android is a computer program or software application designed to run on a Android device , like game app , music app , browser app': 1,
        'Automotive is concerned with self-propelled vehicles or machines': 2,
        'Baby means Baby products that moms will use for their kids, like baby tracker , baby bottles , bottle warmer , baby nipple': 3,
        'Beauty products like Cosmetics are constituted from a mixture of chemical compounds derived from either natural sources or synthetically created ones': 4,
        'Books are long written or printed literary compositions , which tell us good stories or philosophy': 5,
        'CDs and DVDs are digital optical disc data storage formats to store and play digital audio recordings or music , something similar including compact disc (CD), vinyl, audio tape, or another medium': 6,
        'Cell Phones and Accessories refer to mobile phone and some hardware designed for the phone like microphone , headset': 7,
        'Clothing ，Shoes and Jewelry are items worn on the body to protect and comfort the human or for personal adornment': 8,
        'Albums and Digital Music are collections of audio recordings including popular songs and splendid music': 9,
        'Electronics refer to electronic devices, or the part of a piece of equipment that consists of electronic devices': 10,
        'Grocery and Gourmet Food refer to stores primarily engaged in retailing a general range of food products': 11,
        'Health and Personal Care refer to consumer products used in personal hygiene and for beautification': 12,
        'Home and Kitchen refer to something used in Home and Kitchen such as Kitchenware , Tableware , cleaning tools': 13,
        'Kindle Store is an online e-book e-commerce store operated by Amazon as part of its retail website and can be accessed from any Amazon Kindle': 14,
        'Movies and TV is a work of visual art that tells a story and that people watch on a screen or television or a showing of a motion picture especially in a theater': 15,
        'Musical Instruments are devices created or adapted to make musical sounds': 16,
        'Office Products are consumables and equipment regularly used in offices by businesses and other organizations': 17,
        'Patio Lawn and Garden refer to some tools and devices used in garden or lawn': 18,
        'Pet Supplies refer to food or other consumables or tools that will be used when you keep a pet , like dog food , cat treat , pet toy': 19,
        'Sports and Outdoors refer to some tools and sport equipment used in outdoor sports': 20,
        'Tools and Home Improvement refer to hand tools or implements used in the process of renovating a home': 21,
        'Toys and Games are something used in play , usually undertaken for entertainment or fun, and sometimes used as an educational tool.': 22,
        'Video Games or Computer games are electronic games that involves interaction with a user interface or input device to generate visual feedback , which include arcade games , console games , and personal computer (PC) games': 23,
    }

    train_classes = [0, 6, 10, 12, 16, 3, 21, 20, 22, 23]
    val_classes = [13, 18, 2, 14, 1]
    test_classes = [7, 17, 9, 11, 19, 4, 15, 5, 8]

    return train_classes, val_classes, test_classes, label_dict

def _get_fewrel_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
         'place served by transport hub territorial entity or entities served by this transport hub (airport, train station, etc.)': 0,
         'mountain range range or subrange to which the geographical item belongs': 1,
         'religion religion of a person, organization or religious building, or associated with this subject': 2,
         "participating team Like 'Participant' (P710) but for teams. For an event like a cycle race or a football match you can use this property to list the teams and P710 to list the individuals (with 'member of sports team' (P54) as a qualifier for the individuals)": 3,
         'contains administrative territorial entity (list of) direct subdivisions of an administrative territorial entity': 4,
         'head of government head of the executive power of this town, city, municipality, state, country, or other governmental body': 5,
         'country of citizenship the object is a country that recognizes the subject as its citizen': 6,
         'original network network(s) the radio or television show was originally aired on, not including later re-runs or additional syndication': 7,
         'heritage designation heritage designation of a cultural or natural site': 8,
         'performer actor, musician, band or other performer associated with this role or musical work': 9,
         'participant of event a person or an organization was/is a participant in, inverse of P710 or P1923': 10,
         'position held subject currently or formerly holds the object position or public office': 11,
         'has part part of this subject; inverse property of "part of" (P361). See also "has parts of the class" (P2670).': 12,
         'location of formation location where a group or organization was formed': 13,
         'located on terrain feature located on the specified landform. Should not be used when the value is only political/administrative (P131) or a mountain range (P4552).': 14,
         'architect person or architectural firm that designed this building': 15,
         'country of origin country of origin of this item (creative work, food, phrase, product, etc.)': 16,
         'publisher organization or person responsible for publishing books, periodicals, games or software': 17,
         'director director(s) of film, TV-series, stageplay, video game or similar': 18,
         'father male parent of the subject. For stepfather, use "stepparent" (P3448)': 19,
         'developer organisation or person that developed the item': 20,
         'military branch branch to which this military unit, award, office, or person belongs, e.g. Royal Navy': 21,
         'mouth of the watercourse the body of water to which the watercourse drains': 22,
         'nominated for award nomination received by a person, organisation or creative work (inspired from "award received" (Property:P166))': 23,
         'movement literary, artistic, scientific or philosophical movement associated with this person or work': 24,
         'successful candidate person(s) elected after the election': 25,
         'followed by immediately following item in a series of which the subject is a part [if the subject has been replaced, e.g. political offices, use "replaced by" (P1366)]': 26,
         'manufacturer manufacturer or producer of this product': 27,
         'instance of that class of which this subject is a particular example and member (subject typically an individual member with a proper name label); different from P279; using this property as a qualifier is deprecated—use P2868 or P3831 instead': 28,
         'after a work by artist whose work strongly inspired/ was copied in this item': 29,
         'member of political party the political party of which this politician is or has been a member': 30,
         'licensed to broadcast to place that a radio/TV station is licensed/required to broadcast to': 31,
         'headquarters location specific location where an organization\'s headquarters is or has been situated. Inverse property of "occupant" (P466).': 32,
         'sibling the subject has the object as their sibling (brother, sister, etc.). Use "relative" (P1038) for siblings-in-law (brother-in-law, sister-in-law, etc.) and step-siblings (step-brothers, step-sisters, etc.)': 33,
         'instrument musical instrument that a person plays': 34,
         "country sovereign state of this item; don't use on humans": 35,
         'occupation occupation of a person; see also "field of work" (Property:P101), "position held" (Property:P39)': 36,
         'residence the place where the person is or has been, resident': 37,
         'work location location where persons were active': 38,
         'subsidiary subsidiary of a company or organization, opposite of parent organization (P749)': 39,
         'participant person, group of people or organization (object) that actively takes/took part in an event or process (subject).  Preferably qualify with "object has role" (P3831). Use P1923 for participants that are teams.': 40,
         'operator person, profession, or organization that operates the equipment, facility, or service; use country for diplomatic missions': 41,
         'characters characters which appear in this item (like plays, operas, operettas, books, comics, films, TV series, video games)': 42,
         'occupant a person or organization occupying property': 43,
         "genre creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic": 44,
         'operating system operating system (OS) on which a software works or the OS installed on hardware': 45,
         'owned by owner of the subject': 46,
         'platform platform for which a work was developed or released, or the specific platform version of a software product': 47,
         'tributary stream or river that flows into this main stem (or parent) river': 48,
         'winner winner of an event - do not use for awards (use P166 instead), nor for wars or battles': 49,
         'said to be the same as this item is said to be the same as that item, but the statement is disputed': 50,
         'composer person(s) who wrote the music [for lyricist, use "lyrics by" (P676)]': 51,
         'league league in which team or player plays or has played in': 52,
         'record label brand and trademark associated with the marketing of subject music recordings and music videos': 53,
         'distributor distributor of a creative work; distributor for a record label': 54,
         'screenwriter person(s) who wrote the script for subject item': 55,
         'sports season of league or competition property that shows the competition of which the item is a season. Use P5138 for "season of club or team".': 56,
         'taxon rank level in a taxonomic hierarchy': 57,
         'location location of the item, physical object or event is within. In case of an administrative entity use P131. In case of a distinct terrain feature use P706.': 58,
         'field of work specialization of a person or organization; see P106 for the occupation': 59,
         'language of work or name language associated with this creative work (such as books, shows, songs, or websites) or a name (for persons use P103 and P1412)': 60,
         'applies to jurisdiction the item (an institution, law, public office ...) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, ...)': 61,
         "notable work notable scientific, artistic or literary work, or other work of significance among subject's works": 62,
         'located in the administrative territorial entity the item is located on the territory of the following administrative entity. Use P276 (location) for specifying the location of non-administrative places and for items about events': 63,
         'crosses obstacle (body of water, road, ...) which this bridge crosses over or this tunnel goes under': 64,
         'original language of film or TV show language in which a film or a performance work was originally created. Deprecated for written works; use P407 ("language of work or name") instead.': 65,
         'competition class official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion': 66,
         'part of object of which the subject is a part (it\'s not useful to link objects which are themselves parts of other objects already listed as parts of the subject). Inverse property of "has part" (P527, see also "has parts of the class" (P2670)).': 67,
         'sport sport in which the subject participates or belongs to': 68,
         'constellation the area of the celestial sphere of which the subject is a part (from a scientific standpoint, not an astrological one)': 69,
         'position played on team / speciality position or specialism of a player on a team, e.g. Small Forward': 70,
         'located in or next to body of water sea, lake or river': 71,
         "voice type person's voice type. expected values: soprano, mezzo-soprano, contralto, countertenor, tenor, baritone, bass (and derivatives)": 72,
         'follows immediately prior item in a series of which the subject is a part [if the subject has replaced the preceding item, e.g. political offices, use "replaces" (P1365)]': 73,
         'spouse the subject has the object as their spouse (husband, wife, partner, etc.). Use "partner" (P451) for non-married companions': 74,
         'military rank military rank achieved by a person (should usually have a "start time" qualifier), or military rank associated with a position': 75,
         'mother female parent of the subject. For stepmother, use "stepparent" (P3448)': 76,
         'member of organization or club to which the subject belongs. Do not use for membership in ethnic or social groups, nor for holding a position such as a member of parliament (use P39 for that).': 77,
         'child subject has object as biological, foster, and/or adoptive child': 78,
         'main subject primary topic of a work (see also P180: depicts)': 79,
    }


    train_classes = [61, 74, 4, 36, 66, 43, 70, 33, 67, 79, 68, 45, 13, 6, 31, 77, 25, 39, 11, 17, 44, 78, 20, 65, 71, 40, 37, 12, 47, 10, 38, 28, 55, 57, 34, 59, 62, 58, 5, 2, 41, 32, 64, 9, 42, 46, 52, 18, 19, 60, 48, 51, 14, 16, 27, 73, 69, 24, 26, 56, 63, 15, 53, 35, 72]
    val_classes = [29, 30, 54, 23, 49]
    test_classes = [22, 76, 75, 50, 8, 7, 1, 21, 0, 3]

    return train_classes, val_classes, test_classes , label_dict

def _get_huffpost_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'politics government court amendment congressional debate runoff democrat republican Political figures': 0,
        'wellness healthy sport body exercise therapist workout training sleep yoga happiness diet': 1,
        'entertainment enjoy Movie Film and television works and Entertainment star and Entertainment news': 2,
        'travel travelers trip Tourism Tourist destination flight flights airport airlines vacation italian hotel hotels map italy disney': 3,
        'style and beauty beautiful photos adidas fashion clothes dress magazine covers photos makeup looks star': 4,
        'parenting Parents parent mother mom dad uncles kids daughter babies child baby teens': 5,
        'healthy living health care drug medical medicare medicaid disease virus deaths': 6,
        'queer voices LGBT LGBTQ lesbian gay hiv homosexual love straight people trans , nonbinary community gendering transgender Sexual orientation coming out pride ': 7,
        'food foods and drink fruit recipes recipe delicious sandwich pizza chicken wine': 8,
        'business company uber amazon wells fargo bank bankrupt billionaire stock leader ceo lead worker': 9,
        'comedy Interesting thing Gossip jokes hilariously funny jimmy show stephen': 10,
        'sports olympic olympics game team athletes player players bowl winter hockey baseball basketball football soccer gymnastics skate': 11,
        'black voices racist racism police cop white people rapper black men martin luther king': 12,
        'home & living apartment home butler home design furniture bedroom holiday christmas ': 13,
        'parents parenting parent mother mom dad uncles kids child baby babies teens family': 14,
        'the world post korea u.s. china france war attack nuclear missile iran refugees isis egypt syria america referendum ': 15,
        'weddings wedding engagement honeymoon bridal marriage love couples dresses brides bride groom bridesmaid romance newlyweds ring knot guests': 16,
        'women woman female harassment men sexual gender sexist feminism abortions abortion': 17,
        'impact refugees fight hurricane homelessness shelters donate poverty hunger homeless': 18,
        'divorce divorced uncoupling breakup single dating spouse husband wife children marriage married family  infidelity alimony': 19,
        'crime shooting shooter shot hostage crime murder gunman suspect charged arrested arrest inmate police victim': 20,
        'media advertisers media twitter editor journalists journalism news editor journalist newspaper': 21,
        'weird news weird halloween fark quiz': 22,
        'green climate hurricane storm environment environmental climate change wildfire coal animal whale gorilla': 23,
        'world post iran world isis war greece china syrian russia africa crisis america yemen mediterranean poland elections migrants philippines': 24,
        'religion pope muslim meditation ramadan faith church muslims christian religious christians god religion christianity evangelicals catholic jesus': 25,
        'style clothes beauty fashion hair dress makeup prince clothing carpet': 26,
        'science scientists space nasa science earth brain telescope galaxy astronomers': 27,
        'world news reuters president embassy attack australia israel zimbabwe myanmar jerusalem': 28,
        'taste food foods sweets meal foods barbecue salads wine coffee recipes cooking': 29,
        'tech facebook apple google iphone twitter uber microsoft instagram samsung users app encryption android hackers cyber web hackers': 30,
        'money credit tax financial finances lottery investor savings costs buy debt mortgage banks bank money': 31,
        'arts art artist stage ballet music photography photographer nighter theatre dance': 32,
        'fifty midlife retire age care grandma mother retirement aging alzheimer older grandkids childhood': 33,
        'good news selfless kittens dog cat rescue rescued adorable': 34,
        'arts and culture book new artist women museum books broadway history authors sculpture potter': 35,
        'environment week climate animal tigers giraffes animals weather tornado  oil species chemicals': 36,
        'college professors faculty chancellor professor freshman student campus university universities colleges fraternity commencement': 37,
        'latino voices latinos latina immigrants immigration spanish latin mexico mexican hispanic': 38,
        'culture and arts photos image blog artist artists culture gallery exhibition photo theatre photography paintings': 39,
        'education classrooms classroom education learning student students school schools school districts stem educational college teacher teachers teaching': 40
    }

    train_classes = [0, 1, 3, 4, 9, 10, 12, 14, 15, 17, 19, 20, 21, 23, 29, 30, 32, 33, 35, 37]
    val_classes = [2, 13, 18, 22, 34]
    test_classes = [5, 6, 7, 8, 11, 16, 24, 25, 26, 27, 28, 31, 36, 38, 39, 40]


    return train_classes, val_classes, test_classes , label_dict

def _get_reuters_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'acquisition merge If a company or business person makes an acquisition , they buy another company or part of a company': 0,
        'aluminium Aluminium is a lightweight metal used, for example, for making cooking equipment and aircraft parts': 1,
        'trade deficit , current account deficit mean financial situation in the red , shortage , decrease , decline ': 2,
        'cocoa Cocoa is a brown powder made from the seeds of a tropical tree. It is used in making chocolate . It is usually as goods for trade ': 3,
        'coffee Coffee is the roasted beans or powder from which the drink is made': 4,
        'copper Copper is reddish brown metal that is used to make things such as coins and electrical wires': 5,
        'cotton Cotton is a plant and which produces soft fibres used in making cotton cloth': 6,
        'inflation Inflation is a general increase in the prices of goods and services in a country': 7,
        'oil Oil is a smooth , thick liquid that is used as a fuel and for making the parts of machines move smoothly. Oil is found underground': 8,
        'profit A profit is an amount of money that you gain when you are paid more for something than itcost you to make, get, or do it': 9,
        'gdp gnp gross domestic product gross national product In economics, a country GDP is the total value of goods and services produced within a country in a year': 10,
        'gold Gold is a valuable , yellow-coloured metal that is used for making jewellery and ornaments, and as an international currency': 11,
        'grain Grain is a cereal , especially wheat or corn , that has been harvested and is used for food or in trade': 12,
        'rate A rate is the amount of money that is charged for goods or services or ': 13,
        'industrial production Industrial production refers to the production output and the output of industrial establishments and covers sectors such as mining, manufacturing, electricity, gas and steam and air-conditioning': 14,
        'steel Steel is a very strong metal which is made mainly from iron. Steel is used for making many things, for example bridges, buildings, vehicles, and cutlery . It is a important merchandise': 15,
        'unemployment Unemployment is the fact that people who lose jobs, want jobs cannot get them': 16,
        'cattle Cattle are cows and bulls , including animals living in farm': 17,
        'treasury bank the Treasury is the government department that deals with the country finances': 18,
        'money supply Related policies and news on money supply from the central financial department': 19,
        'gas Gas is a substance like air and burns easily. It is used as a fuel for cooking and heating': 20,
        'orange a round sweet fruit that has a thick orange skin and an orange centre , which is goods in trade': 21,
        'reserves foreign reserves , gold and currency reserves , The amount of foreign currency stored by the central bank': 22,
        'retail Retail is the activity of selling products direct to the public, usually in small quantities': 23,
        'rubber  Rubber is a strong, waterproof, elastic substance made from the juice of a tropical tree or produced chemically. It is used for making tyres, boots, and other products': 24,
        'ship a large boat for travelling on water, especially across the sea': 25,
        'sugar Sugar is a sweet substance that is used to make food and drinks sweet. It is usually as goods for trade': 26,
        'tin A tin is a metal container which is filled with food and sealed in order to preserve the food for long periods of time': 27,
        'tariffs A tariff is a tax that a government collects on goods coming into a country': 28,
        'oils and fats tax mean the tax in oils and fats which Promulgated by the European Union': 29,
        'producer price wholesale A producer price index (PPI) is a price index that measures the average changes in prices received by domestic producers for their output.': 30
    }

    train_classes = list(range(15))
    val_classes = list(range(26, 31))
    test_classes = list(range(15, 26))

    return train_classes, val_classes, test_classes, label_dict

def _get_rcv1_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'company development  The latest corporate development trends of large companies ， including new business ， New company ，Cooperation ': 0,
        'economy suit court lawsuit about litigation cases in the economic field, about court decisions, about the latest announcements': 1,
        'Government announces new policy to company The government has promulgated new measures and released new measures, and the central fiscal policy has been adjusted': 2,
        'Stock exchange , company see listing or delisted companies go public , sees listing or companies are delisted , result in stocks rise or fall ( stock trading , stock exchange )': 3,
        'The latest news from banks and companies The news includes a variety of rankings, company financials, a variety of bank data, from central banks and companies around the world': 4,
        "Company performace and financial statements About the company's financial results , the latest fluctuations in profits , up or down ": 5,
        'Enterprise manufacturing costs , productivity , profit Entity manufacturing enterprises manufacturing costs , productivity , profit has undergone some changes': 6,
        'Companies go bankrupt and share prices fall Poor management of the company led to a falling stock price, heavy debt and even bankruptcy': 7,
        'Debt ,  loan ,  credit The company is not doing well and then gets loans, which leads to credit problems and debt problems': 8,
        'Company Stock News About the stock related operations , the company listed ,  launches IPO , stock subscription , stock repurchase , stock decline': 9,
        'buying companies or shares about acquire new companies , buy stock, bid for companies': 10,
        'sell company or Related Property Sell the company, sell the factory, sell stores': 11,
        'privatisation Privatization models , privatisation of companies, privatisation of bonds , privatise public service': 12,
        'Industrial manufacturing, manufacturing output The annual industrial production output of the national manufacturing industry': 13,
        'Emerging technologies , high tech Emerging technologies , high technology , high-end manufacturing , including automotive manufacturing , personal computers , chips , Internet technology , electronic devices': 14,
        'Pharmaceutical manufacturing, pharmaceutical industry, electronic technology New drug testing , new drug manufacturing , new electronic technology , and car airbag safety': 15,
        'Opening new factories and opening new banks, closing port , factories About opening or closing factories, like building new malls, new refineries, closing ports': 16,
        'nan Air fares, bank deposits and loans, telephone services, port freight': 17,
        'Media and Advertising  ad Advertising and Media Digest , ads , commercial ': 18,
        'Contracts and orders New contracts , new supply agreements , new shipping orders , new agreements , new deals': 19,
        'Competition and Monopoly Companies compete with each other, the monopoly position is threatened, launched an anti-monopoly investigation': 20,
        'CEO ， Managers in the enterprise News about corporate executives , Business Managers , corporate presidents ， CEOs , chiefs and boards': 21,
        'Employers and employees , strikes , labor treatment The labor board appealed on behalf of the Detroit strikers, whose employees work for the company': 22,
        'Freight , shipping , air freight News about freight , shipping , ports , transportation , containers': 23,
        'GDP growth and market recovery With GDP growth and change and market recovery, the economy has a bright future': 24,
        'Currency , monetary policy , economic strategy Currency , monetary policy , economic strategy , national banks , economic union': 25,
        'Price change in many fields The prices in many fields,including industry,raw material,commodities etc. rise,except in service and  manufacture input ': 26,
        'Inflation and economic boost Inflation boosts economics and impacts currency policies ': 27,
        'Family deposits increase  Families save more money and spend less than before.': 28,
        'decrease in retail The decrease in retail causes the economic decline in many fields.': 29,
        'Treasury balances Treasury balances , frozen savings ,school budget shortage': 30,
        'Goods purchasing index rise During goods orders ,manufacturing index ,industry pace ,business index': 31,
        'Jobless and unemployment rate Unemployment rate rise or decrease sharply': 32,
        'Foreign investment and fund Foreign investment , cross-boarder fund and refugee aid rise in many countries': 33,
        'trade and economics coorperation between economies Economical trade , investment flow , budget deficit , investment and intellectual property , cooperation with Asian countries.': 34,
        'Housing starts Housing outlook ,  residential alterations and additions , mortgate rates': 35,
        'Economic indicators  Key economic indicators  , import and export volume , trade balance , latest quarterly stats , following figures': 36,
        'News about the economy News conference ,  economic hardship , issue coins': 37,
        'Official notification of the European Union The European Commission makes relevant regulations , Mad Cow Disease , release documents and announcements': 38,
        'EU single monetary policy Many countries are in deficit , most people are against the single monetary policy , budget plans of EU countries': 39,
        'Reform proposals for the future of the EU The benefits can countries get after the reform , thrash out a new treaty , providing a better life for the people of the EU': 40,
        'International political relations of the EU The EU values their political ties': 41,
        'The Middle East peace process Strenthen trade ， hitorical canlendar , Middle East peace process': 42,
        'Disposal of crimes Arrest criminals ， give a ruling': 43,
        'War and peace Military exercises ， weapon preparation , realignment of forces': 44,
        'Dipolomatic relations Boost links and further the peace process': 45,
        'Accident Accidents resulted in casualties': 46,
        'Art and culture The relationship between art,culture and interests': 47,
        'Human and ecology The influence and observation of human beings on ecology': 48,
        'Fashion and design Western fashion relating to brand, era, and region': 49,
        'diseases and medical conditions Different medical and disease conditions across the world': 50,
        'Employment rate and job market The impact of employees on the organization and industry': 51,
        'Death and casualties  Reports about the thoughts of death and life': 52,
        'Political changes around the world There have been changes in cabinet and leadership around the world': 53,
        'Matters relating to women of international importance matters about important women including Mother Teresa, princesses and queens': 54,
        "International religious affairs The Pope's status , attitude , and the state of the church and the church": 55,
        'The exploration of the universe Exploration on board Mir and a partial solar eclipse': 56,
        'International sports events related affairs International sporting events including football , golf , motor racing , tennis and rugby': 57,
        'The development of tourism in various countries The development of tourism in various countries and the characteristic tourist attractions': 58,
        'Opposition forces Guerrillas ， Democratic Party ， separatist': 59,
        'Emergency weather comditions Emergency weather comditions and its harm': 60,
        'Social welfare Raise social welfare , give home to homeless people': 61,
        'Stock markets Ups and downs of the stock market': 62,
        'Government securities bonds Government securities bonds , maturity bonds , national assets': 63,
        'Money Markets Money market shortage,National debts , futures , money rates , bank liquidity': 64,
        'Interest rate changes in some countries interest rate changes differs in countries, some remain , some decrease , some increase': 65,
        'shipping market in some countries Shipping information , cargo carried and related market changes in different countries': 66,
        'international market exchange rate the value of US dollar rises and Japanese currency continues to weaken ,  which may be a victim of the Asian currency crisis': 67,
        'world crop trade market  Information on price changes, imports, exports and new developments of agricultural products in various countries': 68,
        'Precious metal market in some countries London precious metal prices baised to downside ': 69,
        'arrangements of financial trading market in holidays Arrangements of financial exchanges in several countries during holidays': 70
        }

    train_classes = [14, 27, 34, 70, 23, 41, 32, 24, 47, 64, 62, 65, 63, 10, 55, 53, 43, 44, 15, 12, 58, 19, 49, 54, 13,
                     52, 48, 4, 56, 69, 2, 25, 50, 29, 1, 0, 11]
    val_classes = [66, 8, 22, 16, 35, 21, 5, 36, 9, 40]
    test_classes = [51, 46, 28, 17, 39, 18, 6, 38, 59, 7, 67, 37, 57, 20, 30, 3, 31, 42, 68, 60, 26, 45, 61, 33]

    return train_classes, val_classes, test_classes, label_dict


def _load_json(path):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            item = {
                'label': int(row['label']),
                'text': row['text'][:500]  # truncate the text to 500 tokens
            }

            text_len.append(len(row['text']))

            keys = ['head', 'tail', 'ebd_id']
            for k in keys:
                if k in row:
                    item[k] = row[k]

            data.append(item)

        tprint('Class balance:')

        print(label)

        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data


def _read_words(data, class_name_words):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    for example in data:
        words += example['text']
    for example in class_name_words:
        words += example
    return words


def _meta_split(all_data, train_classes, val_classes, test_classes):
    '''
        Split the dataset according to the specified train_classes, val_classes
        and test_classes

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    for example in all_data:
        if example['label'] in train_classes:
            train_data.append(example)
        if example['label'] in val_classes:
            val_data.append(example)
        if example['label'] in test_classes:
            test_data.append(example)

    return train_data, val_data, test_data


def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list


def _data_to_nparray(data, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)


    # compute the max text length
    text_len = np.array([len(e['text']) for e in data])
    max_text_len = max(text_len)

    # initialize the big numpy array by <pad>
    text = vocab.stoi['<pad>'] * np.ones([len(data), max_text_len],
                                     dtype=np.int64)

    del_idx = []
    # convert each token to its corresponding id
    for i in range(len(data)):
        text[i, :len(data[i]['text'])] = [
                vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>']
                for x in data[i]['text']]

        # filter out document with only unk and pad
        if np.max(text[i]) < 2:
            del_idx.append(i)

    vocab_size = vocab.vectors.size()[0]

    text_len, text, doc_label, raw = _del_by_idx(
            [text_len, text, doc_label, raw], del_idx, 0)

    new_data = {
        'text': text,
        'text_len': text_len,
        'label': doc_label,
        'raw': raw,
        'vocab_size': vocab_size,
    }

    return new_data



def _split_dataset(data, finetune_split):
    """
        split the data into train and val (maintain the balance between classes)
        @return data_train, data_val
    """

    # separate train and val data
    # used for fine tune
    data_train, data_val = defaultdict(list), defaultdict(list)

    # sort each matrix by ascending label order for each searching
    idx = np.argsort(data['label'], kind="stable")

    non_idx_keys = ['vocab_size', 'classes2id', 'is_train', 'n_t', 'n_d', 'avg_ebd']
    for k, v in data.items():
        if k not in non_idx_keys:
            data[k] = v[idx]

    # loop through classes in ascending order
    classes, counts = np.unique(data['label'], return_counts=True)
    start = 0
    for label, n in zip(classes, counts):
        mid = start + int(finetune_split * n)  # split between train/val
        end = start + n  # split between this/next class

        for k, v in data.items():
            if k not in non_idx_keys:
                data_train[k].append(v[start:mid])
                data_val[k].append(v[mid:end])

        start = end  # advance to next class

    # convert back to np arrays
    for k, v in data.items():
        if k not in non_idx_keys:
            data_train[k] = np.concatenate(data_train[k], axis=0)
            data_val[k] = np.concatenate(data_val[k], axis=0)

    return data_train, data_val


def load_dataset(args):
    if args.dataset == '20newsgroup':
        train_classes, val_classes, test_classes, label_dict = _get_20newsgroup_classes(args)
    elif args.dataset == 'amazon':
        train_classes, val_classes, test_classes, label_dict = _get_amazon_classes(args)
    elif args.dataset == 'fewrel':
        train_classes, val_classes, test_classes, label_dict = _get_fewrel_classes(args)
    elif args.dataset == 'huffpost':
        train_classes, val_classes, test_classes, label_dict = _get_huffpost_classes(args)
    elif args.dataset == 'reuters':
        train_classes, val_classes, test_classes, label_dict = _get_reuters_classes(args)
    elif args.dataset == 'rcv1':
        train_classes, val_classes, test_classes, label_dict = _get_rcv1_classes(args)
    else:
        raise ValueError(
            'args.dataset should be one of'
            '[20newsgroup, amazon, fewrel, huffpost, reuters, rcv1]')

    assert(len(train_classes) == args.n_train_class)
    assert(len(val_classes) == args.n_val_class)
    assert(len(test_classes) == args.n_test_class)

    tprint('Loading data')
    all_data = _load_json(args.data_path)
    class_names = []
    class_name_words = []
    for ld in label_dict:
        class_name_dic = {}
        class_name_dic['label'] = label_dict[ld]
        class_name_dic['text'] = ld.lower().split()
        class_names.append(class_name_dic)
        class_name_words.append(class_name_dic['text'])

    tprint('Loading word vectors')

    vectors = Vectors(args.word_vector, cache=args.wv_path)
    vocab = Vocab(collections.Counter(_read_words(all_data, class_name_words)), vectors=vectors,
                  specials=['<pad>', '<unk>'], min_freq=5)

    # print word embedding statistics
    wv_size = vocab.vectors.size()
    tprint('Total num. of words: {}, word vector dimension: {}'.format(
        wv_size[0],
        wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(
            torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    tprint(('Num. of out-of-vocabulary words'
           '(they are initialized to zeros): {}').format(num_oov))

    # Split into meta-train, meta-val, meta-test data
    train_data, val_data, test_data = _meta_split(
            all_data, train_classes, val_classes, test_classes)
    tprint('#train {}, #val {}, #test {}'.format(
        len(train_data), len(val_data), len(test_data)))

    # Convert everything into np array for fast data loading
    class_names = _data_to_nparray(class_names, vocab, args)
    train_data = _data_to_nparray(train_data, vocab, args)
    val_data = _data_to_nparray(val_data, vocab, args)
    test_data = _data_to_nparray(test_data, vocab, args)

    train_data['is_train'] = True
    val_data['is_train'] = True
    test_data['is_train'] = True
    # this tag is used for distinguishing train/val/test when creating source pool

    temp_num = np.argsort(class_names['label'])
    class_names['label'] = class_names['label'][temp_num]
    class_names['text'] = class_names['text'][temp_num]
    class_names['text_len'] = class_names['text_len'][temp_num]

    return train_data, test_data, test_data, class_names, vocab

