import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

letters = {"a":0, "b":1, "c":2, "d":3, "e": 4, "f": 5, "g": 6, "h": 7,"i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15,
           "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24, "z": 25, " ":26}
id_to_char = {i: ch for ch, i in letters.items()}

max_length = 12
tensor1 = []
tensor2 = []
names = [
    # Cool / Mythic / Sci-Fi / Fantasy
    "zeiyre", "griffin", "zephyr", "kaelen", "orion", "nyx", "draven", "sorin",
    "cassian", "azriel", "thalor", "phoenix", "cyrus", "evander", "lysander",
    "xander", "zarek", "kieran", "ryker", "jaxon", "stellan", "torin", "aelin",
    "rowan", "briar", "onyx", "atlas", "soren", "valen", "kyros", "zyon",
    "caelum", "dael", "eroz", "falken", "gavriel", "hyperion", "icarus", "jareth",
    "kassius", "lorcan", "malakai", "nevan", "oberon", "paxton", "quillon",
    "rafe", "sylas", "tavian", "ulric", "vance", "wren", "xavion", "yarden", "zev",
    "valkyrie", "nightshade", "silverwind", "stormbringer", "coriolanus", "aurelius",
    "bellatrix", "andromeda", "persephone", "seraphina", "melisandre", "zephyrine",

    # Classic / Normal English
    "john", "charlie", "samuel", "liam", "noah", "oliver", "elijah", "james",
    "william", "benjamin", "lucas", "henry", "emma", "olivia", "ava", "sophia",
    "isabella", "mia", "charlotte", "amelia", "harper", "evelyn", "abigail",
    "emily", "daniel", "matthew", "david", "joseph", "thomas", "charles",
    "caleb", "nathan", "isaac", "ethan", "mason", "logan", "jacob", "michael",
    "grace", "chloe", "victoria", "madison", "scarlett", "aria", "penelope",
    "layla", "riley", "zoey", "nora", "lily", "eleanor", "hannah", "lillian",
    "addison", "aubrey", "stella", "natalie", "zoe", "leah", "hazel", "violet",
    "aurora", "savannah", "audrey", "brooklyn", "bella", "claire", "skylar",
    "christopher", "alexander", "sebastian", "nathaniel", "christian", "jonathan",
    "nicholas", "jeremiah", "cameron", "elizabeth", "gabriella", "samantha",
    "katherine", "genevieve", "mackenzie", "evangeline", "alexandria", "jacqueline",
    "clementine", "marguerite", "montgomery", "fitzgerald", "maximilian",

    # Japanese
    "kenji", "haruki", "ren", "akira", "hiroshi", "takumi", "daiki", "kaito",
    "ryu", "sora", "sakura", "hana", "yuki", "mei", "aoi", "kaede", "hinata",
    "mizuki", "ayaka", "koharu", "shin", "taro", "kenzo", "raiden", "yumi",
    "katsuhiro", "takahashi", "yoshihiro", "mitsuhide", "nobunaga", "amaterasu",
    "tsukuyomi", "kazehaya", "natsuhiko", "shunsuke", "ryunosuke", "sakurako",

    # Nordic / Scandinavian
    "bjorn", "leif", "ivar", "ragnar", "aksel", "freja", "astrid", "sigrid",
    "ingrid", "magne", "torvald", "gunnar", "halvor", "linnea", "alva", "saga",
    "runa", "viggo", "stina", "thorin", "odin", "loki", "falk", "keld",
    "valdemar", "kristoffer", "ragnhild", "hildegard", "brynhildr", "gudmundur",
    "thorstein", "sigmundur", "svanhildr", "ingeborg", "haraldur",

    # Arabic & Persian
    "zayd", "tariq", "malik", "idris", "karim", "amir", "faris", "omar",
    "hamza", "yasir", "amira", "zara", "nadia", "yasmin", "samira", "soraya",
    "darius", "roshan", "zayn", "kalil", "rafiq", "nasir", "leila", "farah",
    "abdulrahman", "muhammad", "iskandar", "shahrazad", "noorjahan", "zulfiqar",
    "sulaiman", "abdullah", "mustapha", "khadijah", "tahmina", "ferdowsi",

    # Slavic
    "dmitri", "nikolai", "alexei", "viktor", "maksim", "artem", "timur",
    "vadim", "pavel", "ruslan", "katya", "mila", "nadya", "anya", "talia",
    "oksana", "zarina", "yelena", "daria", "kira", "ivan", "boris", "lev",
    "vladimir", "stanislav", "vyacheslav", "sviatoslav", "yaroslav", "anastasia",
    "ekaterina", "aleksandra", "elizaveta", "valeriya", "miroslava", "vladislava",

    # Celtic / Irish / Welsh
    "ciaran", "declan", "callum", "ronan", "cormac", "finley", "conor",
    "killian", "seamus", "brennan", "saoirse", "maeve", "aisling", "fiona",
    "niamh", "rhiannon", "gwen", "carys", "seren", "bronwen", "lachlan",
    "finnian", "sullivan", "callahan", "donovan", "morrigan", "gwendolyn",
    "guinevere", "ceallach", "toirdelbhach", "muireann", "caoimhin",

    # Spanish / Italian / Romance
    "mateo", "santiago", "diego", "thiago", "emiliano", "matias", "nicolas",
    "leonardo", "marco", "lorenzo", "camila", "valeria", "ximena", "lucia",
    "elena", "martina", "catalina", "julieta", "renata", "enzo", "luca",
    "dante", "romeo", "alessio", "rafael", "javier", "gael", "emilio",
    "alessandro", "francesco", "valentina", "antonella", "esmeralda",
    "sebastiano", "cristiano", "giovanni", "salvatore", "benedetta", "marcellus",

    # African (Swahili / Yoruba / Zulu / Amharic)
    "amari", "kofi", "kwame", "jabari", "zuri", "imani", "nia", "amina",
    "chidi", "obinna", "tendai", "zuberi", "makena", "ayodele", "taraji",
    "zalika", "simba", "jelani", "azizi", "bakari", "sekou", "tau",
    "oluwaseun", "chimamanda", "babatunde", "oluwatobi", "folashade", "adebayo",
    "nyongesa", "mwangaza", "thembinkosi", "siphiwe", "nomzamo",

    # South Asian / Sanskrit
    "aarav", "vihaan", "aditya", "arjuna", "rohan", "dev", "kabir", "ishaan",
    "reyansh", "vivaan", "ananya", "diya", "priya", "kavya", "meera", "sanya",
    "tara", "isha", "navya", "riya", "kiran", "ashwin", "bodhi", "maya",
    "siddhartha", "abhimanyu", "vikramaditya", "chandrakant", "priyanka",
    "aishwarya", "vasundhara", "devadatta", "jayavardhan", "arundhati",

    # Greek / Ancient
    "apollo", "perseus", "thalia", "calliope", "selene", "iris", "clio",
    "damon", "theron", "lydia", "phoebe", "ariadne", "zeno", "leonidas",
    "hector", "achilles", "ajax", "castor", "cassia", "delphine", "nyssa",
    "agamemnon", "telemachus", "alcibiades", "hippocrates", "cassiopeia",
    "penthesilea", "polyxena", "chrysanthemum", "demetrius", "theophilus"
]

for name in names:
    word = list(name)
    word.append(" ")
    wordl = [26]*13
    for i in range(0,len(name)):
        wordl[i] = letters[word[i]]
        tensor1.append(wordl.copy())
        tensor2.append(letters[word[i+1]])

tensor1 = torch.tensor(tensor1)
tensor2 = torch.tensor(tensor2)

print (tensor1.shape)
print (tensor2.shape)

class Namer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(27,10)
        self.hidden_layer = nn.Linear(130,64)
        self.out_layer = nn.Linear(64,27)
    def forward(self,x):
        x = self.embed_layer(x)
        x = x.view(x.shape[0],-1)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.out_layer(x)
        return x

Mod1 = Namer()
grader = nn.CrossEntropyLoss()
teacher = optim.Adam(Mod1.parameters(), lr = 0.01)

for epoch in range(500):
    Mod1.zero_grad()
    logits = Mod1(tensor1)
    loss = grader(logits, tensor2)
    loss.backward()
    teacher.step()
    if (epoch%100 == 0):
        print (loss.item())

# User Input
while True:
    first_letter = input("Enter ONE letter: ")
    if first_letter == " ":
        break
    wordl = [26] * 13
    wordl[0] = letters[first_letter]
    for i in range(1,12):
        x = torch.tensor([wordl])
        x = Mod1(x)
        x = F.softmax(x,dim=-1)
        x = torch.multinomial(x,num_samples=1).item()
        if x == 26:
            break
        wordl[i] = x
    for i in range(len(wordl)):
        wordl[i] = id_to_char[wordl[i]]
    print("".join(wordl))