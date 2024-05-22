from kandinsky3 import get_T2I_Flash_pipeline
import torch
import os
from PIL import Image, ImageDraw, ImageFont
device_map = torch.device('cuda:0')
dtype_map = {
    'unet': torch.float32,
    'text_encoder': torch.float16,
    'movq': torch.float32,
}

t2i_pipe = get_T2I_Flash_pipeline(
    device_map, dtype_map
)
prompts = [
    "A toy soldier with a bright red uniform.",
    "A toy soldier holding a shiny silver sword.",
    "A toy soldier wearing a tall black hat.",
    "A toy soldier with a blue and gold shield.",
    "A toy soldier marching with a green flag.",
    "A toy soldier standing on a wooden base.",
    "A toy soldier playing a golden trumpet.",
    "A toy soldier with a large blue backpack.",
    "A toy soldier wearing a camouflage uniform.",
    "A toy soldier holding a small lantern.",
    "A toy soldier with a red and white drum.",
    "A toy soldier wearing glasses.",
    "A toy soldier with a bright yellow plume.",
    "A toy soldier standing next to a small cannon.",
    "A toy soldier saluting with a shiny medal on his chest.",
    "A toy soldier with a detailed map in his hand.",
    "A toy soldier holding a long spear.",
    "A toy soldier with a colorful parrot on his shoulder.",
    "A toy soldier wearing a blue cape.",
    "A toy soldier standing on a green hill."
]
prompts = [
    "A car with butterfly wings instead of doors",
    "A car with a tail like a peacock",
    "A car with a fishtail instead of exhaust pipes",
    "A car with giant mushrooms growing on its roof",
    "A car with a dragon's head replacing the front grille",
    "A car with spider legs emerging from its wheels",
    "A car with a unicorn horn protruding from its hood",
    "A car with a mermaid's tail instead of rear lights",
    "A car with flaming wheels like those of a chariot",
    "A car with a chameleon-like ability to change color instantly",
    "A car with vines and flowers entwined around its body",
    "A car with a jellyfish-like canopy instead of a roof",
    "A car with octopus tentacles extending from its sides",
    "A car with a glowing aura around its rims",
    "A car with feathers instead of windshield wipers",
    "A car with a rainbow trail following it as it moves",
    "A car with a rocket engine strapped to its roof",
    "A car with a giant magnifying glass as a sunroof",
    "A car with wings like a bird, ready to take flight",
    "A car with a giant conch shell as its horn"
]
# os.makedirs("Kid", exist_ok=True)
# for i in range(20):
#     res = t2i_pipe("A Image of a Kid", seed=-1)[0]
#     res.save(f"Kid/Kid_{i}.png")
# exit(1)
os.makedirs("cars_1", exist_ok=True)
for prompt in prompts:
    res = t2i_pipe(prompt)[0]
    res.save(f"cars/{prompt}.png")
exit(1)

art_concept_prompts = [
    "A surrealist painting featuring melting clocks and distorted landscapes.",
    "A minimalist sculpture composed of clean lines and geometric shapes.",
    "An abstract artwork characterized by chaotic brushstrokes and vibrant colors.",
    "A hyper-realistic portrait capturing every detail of the subject's face.",
    "A collage artwork incorporating photographs, newspaper clippings, and paint.",
    "A digital illustration featuring futuristic cityscapes and neon lights.",
    "A mixed-media installation combining sculpture, video, and sound.",
    "A classical painting depicting mythological figures and scenes.",
    "A street art mural featuring bold graffiti and political messages.",
    "A conceptual artwork exploring the intersection of technology and nature.",
    "A contemporary sculpture made from found objects and recycled materials.",
    "A stained glass window depicting religious symbols and biblical stories.",
    "A kinetic sculpture that moves and interacts with the viewer.",
    "A landscape painting capturing the beauty of a serene countryside.",
    "A pop art-inspired print featuring celebrities and consumer products.",
    "An installation piece using light and shadow to create immersive environments.",
    "A traditional ink painting inspired by Chinese calligraphy and brushwork.",
    "A sculpture carved from marble with intricate details and textures.",
    "A digital collage combining photography, graphic design, and typography.",
    "A series of abstract paintings exploring color theory and composition.",
    "A performance art piece involving dance, music, and spoken word.",
    "A still life painting featuring everyday objects arranged in a composition.",
    "An art installation made entirely from recycled materials.",
    "A series of portraits exploring themes of identity and diversity.",
    "A mural depicting scenes from local history and culture.",
    "A conceptual artwork using mirrors to distort and reflect the viewer.",
    "A sculpture inspired by natural forms and organic shapes.",
    "A digital animation exploring themes of memory and nostalgia.",
    "A series of photographs documenting everyday life in urban environments.",
    "An abstract sculpture evoking emotions of movement and energy.",
    "A painting inspired by the cosmos and celestial bodies.",
    "A multimedia installation exploring the relationship between humans and technology.",
    "A series of drawings inspired by dreams and the subconscious.",
    "An installation piece using sound and music to create an immersive experience.",
    "A sculpture created from recycled plastic bottles and materials.",
    "A digital artwork combining 3D modeling and animation techniques.",
    "A series of paintings inspired by nature and the environment.",
    "An abstract sculpture exploring the concept of balance and harmony.",
    "A mixed-media collage incorporating textiles, beads, and embroidery.",
    "A conceptual artwork challenging traditional notions of beauty and aesthetics.",
    "A street art mural depicting social and political issues.",
    "A sculpture made from repurposed industrial materials and machinery.",
    "An interactive art installation inviting viewer participation and engagement.",
    "A series of paintings exploring the human condition and emotions.",
    "A digital print created using algorithmic generative art techniques.",
    "A multimedia installation incorporating video projections and performance art.",
    "A sculpture inspired by architecture and urban landscapes.",
    "A series of drawings exploring the human form and anatomy.",
    "An abstract painting using unconventional materials and techniques.",
    "A mixed-media artwork combining painting, collage, and sculpture.",
    "A conceptual artwork exploring the passage of time and temporality.",
    "A mural celebrating cultural diversity and heritage.",
    "A sculpture representing the intersection of nature and technology.",
    "A digital artwork inspired by retro-futurism and science fiction.",
    "A series of photographs exploring themes of identity and self-expression.",
    "An installation piece using light and shadow to create optical illusions.",
    "A kinetic sculpture that responds to changes in its environment.",
    "A series of paintings inspired by mythology and folklore.",
    "A multimedia installation incorporating virtual reality and augmented reality.",
    "A sculpture made from recycled glass and metal.",
    "An abstract artwork exploring patterns and repetition.",
    "A series of drawings inspired by architecture and urban environments.",
    "A digital animation exploring themes of climate change and environmentalism.",
    "A mixed-media collage incorporating elements of nature and the outdoors.",
    "A conceptual artwork challenging societal norms and conventions.",
    "A mural celebrating the beauty of the natural world.",
    "A sculpture inspired by the human figure and anatomy.",
    "An interactive art installation using sensors and technology to respond to viewers.",
    "A series of paintings exploring the concept of memory and nostalgia.",
    "A digital print created using glitch art and digital manipulation techniques.",
    "A multimedia installation incorporating soundscapes and ambient music.",
    "A sculpture representing the fragility of life and the passage of time.",
    "A series of photographs documenting the changing landscape of a city.",
    "An abstract painting evoking emotions of joy and serenity.",
    "A mixed-media collage exploring themes of identity and cultural heritage.",
    "A conceptual artwork challenging perceptions of reality and illusion.",
    "A mural depicting scenes from local folklore and mythology.",
    "A sculpture made from natural materials such as wood and stone.",
    "An interactive art installation inviting viewers to create and participate.",
    "A series of paintings inspired by dreams and the subconscious mind.",
    "A digital animation exploring themes of isolation and connection.",
    "A multimedia installation incorporating elements of performance and theater.",
    "A sculpture representing the beauty of the natural world and biodiversity.",
    "A series of drawings inspired by the natural world and wildlife.",
    "An abstract artwork exploring the concept of infinity and eternity.",
    "A mural celebrating the resilience and strength of communities.",
    "A sculpture made from recycled paper and cardboard.",
    "An interactive art installation using light and projection mapping.",
    "A series of paintings inspired by the colors and textures of the desert.",
    "A mixed-media collage exploring themes of memory and identity.",
    "A conceptual artwork challenging perceptions of beauty and ugliness.",
    "A mural depicting scenes from local history and folklore.",
    "A sculpture representing the interconnectedness of all living beings.",
    "An interactive art installation inviting viewers to explore and interact.",
    "A series of photographs documenting everyday life in rural communities.",
    "An abstract painting evoking emotions of sorrow and loss.",
    "A digital print created using fractal art and mathematical algorithms."
]
os.makedirs("art_concept", exist_ok=True)
for prompt in art_concept_prompts:
    res = t2i_pipe(prompt)[0]
    res.save(f"art_concept/{prompt}.png")

sunrise_scenes_prompts = [
    "A tranquil beach at sunrise, with the sky painted in hues of orange and pink.",
    "A misty forest as the sun rises, casting long shadows through the trees.",
    "A serene mountain peak bathed in the warm light of the rising sun.",
    "A peaceful countryside scene at dawn, with the first light illuminating the fields.",
    "A city skyline at sunrise, with skyscrapers silhouetted against the colorful sky.",
    "A desert landscape as the sun breaks over the horizon, casting a golden glow.",
    "A tranquil lake at sunrise, with mist rising from the water's surface.",
    "A coastal village waking up to the first light of dawn, with fishing boats heading out to sea.",
    "A meadow in bloom at sunrise, with flowers catching the first rays of sunlight.",
    "A snow-covered forest at dawn, with the sky turning pink as the sun rises.",
    "A river winding through the countryside at sunrise, with fog hovering above the water.",
    "A vineyard at sunrise, with rows of grapevines bathed in golden light.",
    "A farm at dawn, with fields stretching to the horizon and the sun rising behind a barn.",
    "A tropical beach at sunrise, with palm trees swaying in the early morning breeze.",
    "A canyon at dawn, with the first light of day painting the rock formations in warm tones.",
    "A village nestled in the mountains at sunrise, with smoke rising from chimneys.",
    "A park at dawn, with dew glistening on the grass and birds singing in the trees.",
    "A river delta at sunrise, with fog rolling in from the water and mangrove trees silhouetted against the sky.",
    "A hilltop overlooking the ocean at sunrise, with waves crashing against the shore below.",
    "A field of sunflowers at dawn, with the flowers turning to face the rising sun.",
    "A desert oasis at sunrise, with palm trees casting long shadows in the early morning light.",
    "A coastal cliff at dawn, with seagulls circling overhead and waves crashing against the rocks below.",
    "A village in the mountains waking up to the first light of day, with smoke rising from chimneys.",
    "A forest clearing at sunrise, with a stream babbling softly in the early morning light.",
    "A valley at dawn, with mist rising from the forest below and the sun just beginning to crest the mountains.",
    "A medieval castle at sunrise, with the first light of day illuminating the ancient stone walls.",
    "A beachside boardwalk at dawn, with the first joggers of the day out for their morning run.",
    "A tropical island at sunrise, with palm trees swaying in the gentle breeze and waves lapping at the shore.",
    "A hillside vineyard at dawn, with rows of grapevines stretching into the distance and the sun rising over the horizon.",
    "A forest glade at sunrise, with dew-covered ferns catching the first light of day.",
    "A rural village at dawn, with smoke rising from chimneys and roosters crowing in the distance.",
    "A riverbank at sunrise, with fishermen casting their lines into the water and birds singing in the trees.",
    "A mountain pass at dawn, with mist swirling around the peaks and the first light of day breaking over the horizon.",
    "A coastal town waking up to the first light of dawn, with fishing boats returning to harbor.",
    "A field of lavender at sunrise, with the purple flowers bathed in golden light.",
    "A lakeside cabin at dawn, with smoke rising from the chimney and a rowboat moored at the dock.",
    "A canyon at sunrise, with the first light of day casting dramatic shadows on the rock walls.",
    "A rural farmhouse at dawn, with fields of wheat stretching to the horizon and the sun rising behind the barn.",
    "A beach at sunrise, with seashells scattered along the shore and the sound of waves crashing in the distance.",
    "A forest pond at dawn, with mist rising from the water and birds beginning to stir in the trees.",
    "A mountain lake at sunrise, with the still waters reflecting the pink and gold hues of the sky.",
    "A coastal village at dawn, with fishermen heading out to sea and seagulls circling overhead.",
    "A wheat field at sunrise, with the golden stalks swaying in the breeze and the first light of day illuminating the landscape.",
    "A river at dawn, with fog hovering above the water and the sun just beginning to peek over the horizon.",
    "A hilltop monastery at sunrise, with monks gathering for morning prayers as the first light of day fills the sky.",
    "A tropical rainforest at dawn, with the sounds of birdsong echoing through the trees and mist hanging in the air.",
    "A coastal lighthouse at sunrise, with the beacon casting a warm glow over the ocean.",
    "A village square at dawn, with market stalls being set up and the smell of fresh bread wafting through the air.",
    "A desert canyon at sunrise, with the first light of day illuminating the red rock formations.",
    "A rural barn at dawn, with fields of corn stretching to the horizon and the sun rising behind the silo.",
    "A beachside café at sunrise, with tables set up on the sand and the smell of coffee drifting on the breeze.",
    "A mountain ridge at dawn, with the first light of day painting the peaks in shades of pink and purple.",
    "A coastal marsh at sunrise, with fog rolling in from the ocean and birds wading in the shallows.",
    "A medieval town at dawn, with narrow cobblestone streets and the first light of day glinting off the rooftops.",
    "A forest waterfall at sunrise, with the water cascading over rocks and sunlight filtering through the trees.",
    "A vineyard at dawn, with workers harvesting grapes and the sun just beginning to rise over the horizon.",
    "A river valley at sunrise, with mist rising from the water and the sound of birdsong filling the air.",
    "A coastal promenade at dawn, with joggers out for their morning run and seagulls wheeling overhead.",
    "A mountain meadow at sunrise, with wildflowers blooming and the first light of day painting the landscape in pastel hues.",
    "A rural church at dawn, with the sound of church bells ringing out across the countryside.",
    "A forest trail at sunrise, with shafts of golden light filtering through the trees and dappling the forest floor.",
    "A seaside village at dawn, with fishermen mending their nets and seagulls squawking in the harbor.",
    "A hilltop vineyard at sunrise, with rows of grapevines stretching into the distance and the sun just beginning to rise over the horizon.",
    "A riverbank at dawn, with mist rising from the water and the sound of birdsong filling the air.",
    "A coastal promenade at dawn, with joggers out for their morning run and seagulls wheeling overhead.",
    "A mountain meadow at sunrise, with wildflowers blooming and the first light of day painting the landscape in pastel hues.",
    "A rural church at dawn, with the sound of church bells ringing out across the countryside.",
    "A forest trail at sunrise, with shafts of golden light filtering through the trees and dappling the forest floor.",
    "A seaside village at dawn, with fishermen mending their nets and seagulls squawking in the harbor.",
    "A hilltop vineyard at sunrise, with rows of grapevines stretching into the distance and the sun just beginning to rise over the horizon."
]

os.makedirs("sunrise_scene", exist_ok=True)
for prompt in sunrise_scenes_prompts:
    res = t2i_pipe(prompt)[0]
    res.save(f"sunrise_scene/{prompt}.png")

child_sunrise_scenes_prompts = [
    "A child playing on a tranquil beach at sunrise, building sandcastles as the sky turns pink and gold.",
    "A young girl running through a misty forest at dawn, laughing as she chases after fireflies.",
    "A boy standing on a mountain peak at sunrise, arms outstretched as he welcomes the new day.",
    "A child sitting in a meadow at dawn, picking wildflowers and watching the sun rise over the horizon.",
    "A young boy fishing by a river at sunrise, his face lit up with excitement as he reels in his first catch.",
    "A girl watching the sunrise from her bedroom window, wrapped in a cozy blanket and sipping hot chocolate.",
    "A boy and his dog playing in a wheat field at dawn, running through the golden stalks as the sun rises.",
    "A child gazing out over a mist-covered lake at sunrise, lost in wonder at the beauty of the world.",
    "A girl riding her bike along a coastal path at dawn, the sea breeze blowing through her hair as the sun rises over the ocean.",
    "A young boy sitting on a hilltop at sunrise, watching as the first rays of light illuminate the countryside below.",
    "A child exploring a forest trail at dawn, marveling at the dew-covered leaves and the birdsong filling the air.",
    "A girl and her father watching the sunrise from a hilltop, their faces bathed in the warm glow of the morning light.",
    "A boy skipping stones across a river at sunrise, his laughter echoing through the quiet morning air.",
    "A child playing with a kite on a beach at dawn, the colorful fabric dancing in the early morning breeze.",
    "A young girl picking apples in an orchard at sunrise, the sweet scent of fruit filling the air as the sun rises.",
    "A boy and his sister watching the sunrise from their treehouse, bundled up in blankets and sipping hot cocoa.",
    "A child splashing in a puddle at sunrise, giggling as the first light of day reflects off the water.",
    "A girl and her dog walking along a forest path at dawn, the crisp morning air filling their lungs as the sun rises.",
    "A boy and his grandfather watching the sunrise from a dock, their fishing poles forgotten as they take in the beauty of the morning.",
    "A child sitting by a campfire at dawn, roasting marshmallows as the sun rises over the horizon.",
    "A young girl playing with a toy boat in a pond at sunrise, the water reflecting the pink and gold hues of the sky.",
    "A boy and his mother watching the sunrise from a hilltop, their arms wrapped around each other as they share a moment of quiet reflection.",
    "A child running through a field of wildflowers at dawn, laughter bubbling up as petals catch in their hair.",
    "A girl and her father hiking along a mountain trail at sunrise, their footsteps crunching on the dew-covered grass.",
    "A boy sitting on a dock at dawn, fishing rod in hand as he waits for the first bite of the day.",
    "A child swinging on a tire swing at sunrise, the cool morning air rushing past their face as they soar through the air.",
    "A young girl playing with a puppy in a meadow at dawn, the golden light of the rising sun warming their faces.",
    "A boy and his friends racing each other along the beach at sunrise, the sand flying up behind them as they run.",
    "A child lying in a hammock at dawn, swaying gently in the breeze as they watch the sun rise over the treetops.",
    "A girl and her siblings exploring a forest at sunrise, their laughter echoing through the trees as they search for hidden treasures.",
    "A boy and his grandfather sitting on a porch swing at dawn, sipping hot tea as they watch the world wake up.",
    "A child playing with a soccer ball in a park at sunrise, the grass still damp with dew as they kick the ball back and forth.",
    "A young girl flying a kite on a hillside at dawn, the colorful fabric soaring high above her head.",
    "A boy and his sister picking berries in a field at sunrise, their fingers stained purple as they fill their baskets with fruit.",
    "A child and their parents watching the sunrise from a hot air balloon, the world spread out below them as they float silently through the sky.",
    "A girl skipping stones across a pond at dawn, her face lighting up with delight as each stone skips across the water.",
    "A boy and his friends playing tag in a park at sunrise, the sound of their laughter filling the air as they chase each other through the grass.",
    "A child and their grandmother sitting on a porch swing at dawn, wrapped in blankets as they share stories and watch the sun rise.",
    "A young girl and her brother exploring a forest at sunrise, their pockets filled with treasures they've found along the way.",
    "A boy and his father fishing in a river at dawn, the sound of rushing water and birdsong filling the air as they cast their lines.",
    "A child playing with a frisbee on a beach at sunrise, the cool sand squishing between their toes as they run and jump.",
    "A girl and her mother riding bikes along a coastal path at dawn, the salty breeze ruffling their hair as they pedal.",
    "A boy and his sister building a sandcastle on a beach at sunrise, the waves lapping at the shore as they work.",
    "A child and their parents watching the sunrise from a mountaintop, the world spread out below them in all its glory.",
    "A young girl playing with a kite in a field at dawn, the colorful fabric dancing in the early morning breeze.",
    "A boy and his friends exploring a forest at sunrise, their voices echoing through the trees as they search for adventure.",
    "A child and their grandparents watching the sunrise from a cabin porch, wrapped in blankets and sipping hot cocoa as they welcome the day.",
    "A girl and her brother picking flowers in a meadow at dawn, the scent of blooms filling the air as they gather bouquets.",
    "A boy and his father hiking along a mountain trail at sunrise, the sound of birdsong and rushing water accompanying their footsteps.",
    "A child and their friends playing in a park at dawn, the swings creaking and the slide slippery with dew as they race and chase.",
    "A young girl and her sister skipping stones across a pond at sunrise, their laughter ringing out across the water.",
    "A boy and his mother watching the sunrise from a hilltop, their faces turned towards the sky as they greet the new day.",
    "A child and their father fishing in a lake at dawn, the first light of day casting long shadows across the water.",
    "A girl playing with a toy boat in a pond at sunrise, the ripples from her toy reflecting the colors of the sky.",
    "A boy and his friends building a fort in a field at dawn, their imaginations running wild as they construct their hideaway.",
    "A child and their grandparents watching the sunrise from a porch swing, wrapped in blankets as they share stories and laughter.",
    "A young girl and her brother exploring a forest at sunrise, their pockets filled with treasures and their faces shining with excitement.",
    "A boy and his father riding bikes along a coastal path at dawn, the sea breeze blowing through their hair as they pedal.",
    "A child and their mother watching the sunrise from a hot air balloon, the world spread out below them as they float through the sky.",
    "A girl and her friends playing tag in a park at sunrise, the grass still damp with dew as they run and laugh.",
    "A boy and his sister flying kites on a hillside at dawn, the colorful fabric soaring high above them as they run and jump.",
    "A child and their grandparents watching the sunrise from a mountaintop, the world spread out below them in all its glory.",
    "A young girl and her brother fishing in a river at dawn, the sound of rushing water and birdsong filling the air as they cast their lines.",
    "A boy and his friends riding bikes along a coastal path at dawn, the salty breeze ruffling their hair as they pedal.",
    "A child and their parents watching the sunrise from a beach at dawn, the waves crashing against the shore as they greet the new day.",
    "A girl and her mother picking flowers in a meadow at sunrise, the dew-covered petals shining in the early morning light.",
    "A boy and his father playing catch in a park at dawn, the first light of day casting long shadows across the grass.",
    "A child and their grandparents watching the sunrise from a porch swing, wrapped in blankets and sipping hot cocoa as they share stories and laughter.",
    "A young girl and her brother exploring a forest at sunrise, their pockets filled with treasures and their faces shining with excitement.",
    "A boy and his mother riding bikes along a coastal path at dawn, the sea breeze blowing through their hair as they pedal.",
    "A child and their father watching the sunrise from a hot air balloon, the world spread out below them as they float through the sky.",
    "A girl and her friends playing tag in a park at sunrise, the grass still damp with dew as they run and laugh.",
    "A boy and his sister flying kites on a hillside at dawn, the colorful fabric soaring high above them as they run and jump.",
    "A child and their grandparents watching the sunrise from a mountaintop, the world spread out below them in all its glory.",
    "A young girl and her brother fishing in a river at dawn, the sound of rushing water and birdsong filling the air as they cast their lines.",
    "A boy and his friends riding bikes along a coastal path at dawn, the salty breeze ruffling their hair as they pedal.",
    "A child and their parents watching the sunrise from a beach at dawn, the waves crashing against the shore as they greet the new day.",
    "A girl and her mother picking flowers in a meadow at sunrise, the dew-covered petals shining in the early morning light.",
    "A boy and his father playing catch in a park at dawn, the first light of day casting long shadows across the grass."
]

os.makedirs("child_sunrise_scene", exist_ok=True)
for prompt in child_sunrise_scenes_prompts:
    res = t2i_pipe(prompt)[0]
    res.save(f"child_sunrise_scene/{prompt}.png")

prompts = [
    "A curly-haired child blowing bubbles in the park.",
    "A freckle-faced kid building a sandcastle on the beach.",
    "A toddler with pigtails chasing butterflies in a garden.",
    "A boy with glasses reading a book under a tree.",
    "A little girl with braids eating ice cream at a carnival.",
    "A child with a missing tooth playing with a puppy in the backyard.",
    "A redhead kid flying a kite in a windy field.",
    "A child with a baseball cap playing catch with their dad.",
    "A girl with braces riding a bicycle down the street.",
    "A boy with a band-aid on his knee climbing a tree.",
    "A child with a backpack waiting for the school bus.",
    "A kid with a cowboy hat riding a toy horse in the living room.",
    "A girl with a bow in her hair jumping rope on the sidewalk.",
    "A little boy with a teddy bear having a tea party.",
    "A child with a superhero cape pretending to fly.",
    "A boy with a skateboard doing tricks at the skatepark.",
    "A girl with a hula hoop dancing in the backyard.",
    "A child with a pirate costume digging for treasure in the sandbox.",
    "A kid with a chef's hat baking cookies in the kitchen.",
    "A child with a backpack walking to school in the rain.",
    "A girl with a flower crown picking wildflowers in a field.",
    "A boy with a paper airplane standing on top of a hill.",
    "A child with a snorkel exploring underwater in the pool.",
    "A kid with a magician's hat performing tricks on stage.",
    "A child with a football helmet scoring a touchdown.",
    "A girl with a fairy costume sprinkling glitter in the forest.",
    "A boy with a magnifying glass searching for bugs in the garden.",
    "A child with a space helmet pretending to walk on the moon.",
    "A kid with a detective hat solving a mystery.",
    "A child with a firefighter hat rescuing a stuffed animal from a tree.",
    "A girl with a tiara riding a carousel at the amusement park.",
    "A boy with a conductor's hat riding a toy train around the room.",
    "A child with a construction hat building a sandcastle on the beach.",
    "A kid with a chef's hat stirring a pot on the stove.",
    "A child with a doctor's coat giving a checkup to a teddy bear.",
    "A girl with a flower crown playing with a ladybug in the garden.",
    "A boy with a superhero mask leaping over tall buildings.",
    "A child with a painter's smock creating a masterpiece on canvas.",
    "A kid with a pirate hat steering a toy ship in a bathtub.",
    "A child with a baseball glove catching a fly ball in the outfield.",
    "A girl with a princess gown twirling in a ballroom.",
    "A boy with a knight's helmet jousting with a cardboard lance.",
    "A child with a scuba mask diving into the swimming pool.",
    "A kid with a police hat directing traffic on the sidewalk.",
    "A child with a cowboy hat lassoing a stuffed animal.",
    "A girl with a fairy wand casting spells in the garden.",
    "A boy with a race car driver's helmet speeding around a track.",
    "A child with a safari hat spotting animals on a jungle adventure.",
    "A kid with a scientist's goggles conducting experiments in a lab.",
    "A child with a teacher's pointer giving a lesson to stuffed animals.",
    "A girl with a witch's hat stirring a cauldron of potion.",
    "A boy with a wizard's hat waving a magic wand.",
    "A child with a chef's hat flipping pancakes in a kitchen.",
    "A kid with a doctor's stethoscope listening to a heartbeat.",
    "A child with a firefighter hat spraying a hose.",
    "A girl with a painter's palette mixing colors on a canvas.",
    "A boy with a police hat directing traffic with a whistle.",
    "A child with a cowboy hat riding a hobby horse.",
    "A kid with a superhero mask flexing muscles.",
    "A child with a pirate hat digging for buried treasure in a sandbox.",
    "A girl with a princess tiara riding a unicorn.",
    "A boy with a knight's helmet guarding a castle.",
    "A child with a scuba mask swimming with dolphins in the ocean.",
    "A kid with a safari hat observing giraffes on a savanna.",
    "A child with a scientist's goggles examining specimens under a microscope.",
    "A girl with a teacher's glasses reading a book to her toys.",
    "A boy with a wizard's hat casting spells with a wand.",
    "A child with a chef's hat decorating cupcakes with icing.",
    "A kid with a doctor's coat bandaging a teddy bear's paw.",
    "A child with a firefighter hat rescuing a cat from a tree.",
    "A girl with a painter's smock painting a mural on a wall.",
    "A boy with a police hat patrolling the neighborhood on a bike.",
    "A child with a cowboy hat herding toy cows in a makeshift corral.",
    "A kid with a superhero cape striking a heroic pose.",
    "A child with a pirate hat steering a toy ship through rough waters.",
    "A girl with a princess gown dancing at a royal ball.",
    "A boy with a knight's helmet battling a dragon with a cardboard sword.",
    "A child with a scuba mask exploring a coral reef.",
    "A kid with a safari hat tracking a lion on the plains.",
    "A child with a scientist's goggles mixing chemicals in test tubes.",
    "A girl with a teacher's glasses writing on a chalkboard.",
    "A boy with a wizard's hat studying spellbooks in a library.",
    "A child with a chef's hat rolling dough for homemade pizza.",
    "A kid with a doctor's coat giving a checkup to a doll.",
    "A child with a firefighter hat climbing a ladder to rescue a toy.",
    "A girl with a painter's palette painting a landscape in the park.",
    "A boy with a police hat arresting a pretend robber.",
    "A child with a cowboy hat riding a stick horse in a rodeo.",
    "A kid with a superhero mask scaling a tall building.",
    "A child with a pirate hat burying treasure in the backyard.",
    "A girl with a princess tiara riding a horse-drawn carriage.",
    "A boy with a knight's helmet defending a castle from invaders.",
    "A child with a scuba mask swimming with sharks in the deep sea.",
    "A kid with a safari hat photographing elephants on a safari.",
    "A child with a scientist's goggles conducting experiments in a laboratory.",
    "A girl with a teacher's glasses reading a story to her dolls.",
    "A boy with a wizard's hat brewing potions in a cauldron.",
    "A child with a chef's hat baking cookies in the oven.",
    "A kid with a doctor's coat examining a teddy bear's heartbeat.",
    "A child with a firefighter hat putting out a pretend fire with a hose.",
    "A girl with a painter's smock painting a portrait in an art studio.",
    "A boy with a police hat writing a ticket for a pretend violation.",
    "A child with a cowboy hat riding a toy bull in a rodeo.",
    "A kid with a superhero cape flying through the sky.",
    "A child with a pirate hat navigating a toy ship through stormy seas.",
    "A girl with a princess gown attending a royal banquet.",
    "A boy with a knight's helmet rescuing a damsel in distress.",
    "A child with a scuba mask exploring a sunken shipwreck.",
    "A kid with a safari hat observing zebras on an African savanna.",
    "A child with a scientist's goggles studying a bubbling beaker.",
    "A girl with a teacher's glasses teaching her toys a lesson.",
    "A boy with a wizard's hat casting spells in a magical forest.",
    "A child with a chef's hat decorating a birthday cake with frosting.",
    "A kid with a doctor's coat bandaging a stuffed animal's injury.",
    "A child with a firefighter hat rescuing a toy from a pretend fire.",
    "A girl with a painter's palette painting a sunset over the ocean.",
    "A boy with a police hat directing traffic with a toy whistle.",
    "A child with a cowboy hat herding imaginary cattle on the range.",
    "A kid with a superhero mask stopping a runaway train.",
    "A child with a pirate hat digging for buried treasure on a desert island.",
    "A girl with a princess tiara dancing at a grand ball.",
    "A boy with a knight's helmet battling a fierce dragon in a castle.",
    "A child with a scuba mask swimming with colorful fish in a coral reef.",
    "A kid with a safari hat photographing lions on an African plain.",
    "A child with a scientist's goggles examining specimens under a microscope.",
    "A girl with a teacher's glasses teaching her toys how to read.",
    "A boy with a wizard's hat studying ancient spellbooks in a secret library.",
    "A child with a chef's hat baking cupcakes for a birthday party.",
    "A kid with a doctor's coat giving a checkup to a stuffed animal patient.",
    "A child with a firefighter hat rescuing a toy from a burning building.",
    "A girl with a painter's smock painting a mural on a city wall.",
    "A boy with a police hat patrolling the neighborhood with a toy walkie-talkie.",
    "A child with a cowboy hat riding a pretend horse in a make-believe rodeo.",
    "A kid with a superhero cape saving the day from an evil villain.",
    "A child with a pirate hat steering a toy ship through treacherous waters.",
    "A girl with a princess gown attending a royal tea party with her toys.",
    "A boy with a knight's helmet defending a castle from imaginary invaders.",
    "A child with a scuba mask exploring a mysterious underwater cave.",
    "A kid with a safari hat observing elephants on an African safari.",
    "A child with a scientist's goggles conducting experiments in a laboratory.",
    "A girl with a teacher's glasses reading a story to her stuffed animal class.",
    "A boy with a wizard's hat casting spells with a magic wand in a enchanted forest.",
    "A child with a chef's hat baking cookies in a toy oven.",
    "A kid with a doctor's coat bandaging a teddy bear's injury with care.",
    "A child with a firefighter hat rescuing a kitten from a tall tree.",
    "A girl with a painter's palette painting a colorful rainbow in the sky.",
    "A boy with a police hat directing traffic with a toy stop sign.",
    "A child with a cowboy hat herding imaginary sheep on a pretend ranch.",
    "A kid with a superhero mask stopping a runaway train with sheer bravery.",
    "A child with a pirate hat digging for buried treasure on a deserted island.",
    "A girl with a princess tiara dancing gracefully at a magical ball.",
    "A boy with a knight's helmet bravely defending a fortress from a fierce dragon.",
    "A child with a scuba mask swimming with playful dolphins in a crystal-clear sea.",
    "A kid with a safari hat photographing majestic lions on an African savanna.",
    "A child with a scientist's goggles examining intricate specimens under a microscope.",
    "A girl with a teacher's glasses patiently teaching her stuffed animals to read.",
    "A boy with a wizard's hat practicing spells from an ancient spellbook in a hidden chamber.",
    "A child with a chef's hat baking delicious cupcakes for a birthday celebration.",
    "A kid with a doctor's coat tenderly caring for a stuffed animal patient in need.",
    "A child with a firefighter hat bravely rescuing a puppy from a burning building.",
    "A girl with a painter's smock joyfully painting a vibrant mural on a city wall.",
    "A boy with a police hat diligently patrolling the neighborhood with a toy flashlight.",
    "A child with a cowboy hat herding make-believe cattle through a pretend prairie.",
    "A kid with a superhero cape fearlessly defending the city from a menacing villain.",
    "A child with a pirate hat embarking on an epic adventure to find buried treasure.",
    "A girl with a princess gown twirling gracefully at a magical tea party with her toys.",
    "A boy with a knight's helmet valiantly battling a ferocious dragon to save a kingdom.",
    "A child with a scuba mask exploring the wonders of an underwater world teeming with life.",
    "A kid with a safari hat capturing breathtaking photos of exotic animals on an African safari.",
    "A child with a scientist's goggles carefully conducting experiments in a state-of-the-art laboratory.",
    "A girl with a teacher's glasses enthusiastically teaching her stuffed animal students valuable lessons.",
    "A boy with a wizard's hat mastering powerful spells from an ancient book of magic.",
    "A child with a chef's hat proudly presenting a batch of freshly baked cookies from the oven.",
    "A kid with a doctor's coat compassionately tending to a beloved stuffed animal in need of medical attention.",
    "A child with a firefighter hat courageously rescuing a kitten from a burning tree.",
    "A girl with a painter's smock happily creating a masterpiece on a canvas with bright colors.",
    "A boy with a police hat bravely patrolling the neighborhood with a toy badge and whistle.",
    "A child with a cowboy hat rounding up imaginary cattle on a wild west adventure.",
    "A kid with a superhero cape fearlessly defending the city from evildoers with superhuman strength.",
    "A child with a pirate hat setting sail on a daring quest to discover hidden treasure.",
    "A girl with a princess tiara attending a royal ball with her beloved stuffed animal friends.",
    "A boy with a knight's helmet gallantly rescuing a fair maiden from a fire-breathing dragon.",
    "A child with a scuba mask exploring the mysteries of the deep sea alongside colorful marine life.",
    "A kid with a safari hat photographing majestic wildlife on an exciting safari adventure.",
    "A child with a scientist's goggles conducting groundbreaking experiments in a high-tech laboratory.",
    "A girl with a teacher's glasses imparting valuable knowledge to her attentive stuffed animal students.",
    "A boy with a wizard's hat casting powerful spells from an ancient grimoire in a mystical forest.",
    "A child with a chef's hat proudly presenting a delectable feast of homemade treats from the kitchen.",
    "A kid with a doctor's coat compassionately caring for a sick stuffed animal with tender love and care.",
    "A child with a firefighter hat bravely rescuing a stranded kitten from a towering tree.",
    "A girl with a painter's smock joyfully painting a colorful mural on a city wall with vibrant hues.",
    "A boy with a police hat diligently patrolling the neighborhood with a toy police car and walkie-talkie.",
    "A child with a cowboy hat herding imaginary cattle on a make-believe ranch in the wild west.",
    "A kid with a superhero cape fearlessly protecting the city from villains with superhuman strength.",
    "A child with a pirate hat embarking on a daring adventure to uncover hidden treasure on a remote island.",
    "A girl with a princess tiara attending a magical ball with her favorite stuffed animals as honored guests.",
    "A boy with a knight's helmet valiantly rescuing a fair maiden from the clutches of an evil sorcerer.",
    "A child with a scuba mask exploring the wonders of an underwater world filled with colorful coral reefs and exotic fish.",
    "A kid with a safari hat photographing majestic lions and elephants on an exciting safari adventure in Africa.",
    "A child with a scientist's goggles conducting groundbreaking experiments in a state-of-the-art laboratory filled with bubbling beakers and whirring machinery.",
    "A girl with a teacher's glasses imparting valuable knowledge to her attentive stuffed animal students during a pretend classroom lesson.",
    "A boy with a wizard's hat casting powerful spells from an ancient book of magic in a mystical forest filled with towering trees and magical creatures.",
    "A child with a chef's hat proudly presenting a delicious assortment of freshly baked cookies, cakes, and pastries from the kitchen oven.",
    "A kid with a doctor's coat tenderly caring for a sick stuffed animal patient with a gentle touch and a warm smile.",
    "A child with a firefighter hat bravely rescuing a frightened kitten from the top of a tall tree with a sturdy ladder and a strong grip.",
    "A girl with a painter's smock joyfully painting a vibrant mural on a city wall with bold strokes and bright colors.",
    "A boy with a police hat patrolling the neighborhood streets with a toy police car and a loud siren to keep the peace and protect the community.",
    "A child with a cowboy hat herding make-believe cattle on a pretend ranch in the wide-open plains of the wild west.",
    "A kid with a superhero cape fearlessly defending the city from evildoers with superhuman strength and lightning-fast reflexes.",
    "A child with a pirate hat embarking on an epic journey across the high seas to discover hidden treasure on a remote tropical island.",
    "A girl with a princess tiara attending a glamorous royal ball with her favorite stuffed animals as honored guests.",
    "A boy with a knight's helmet gallantly rescuing a fair maiden from the clutches of an evil dragon in a faraway kingdom.",
    "A child with a scuba mask exploring the wonders of an underwater world teeming with colorful fish, coral reefs, and exotic sea creatures.",
    "A kid with a safari hat photographing majestic lions, elephants, and giraffes on an exciting safari adventure in the African savanna.",
    "A child with a scientist's goggles conducting groundbreaking experiments in a state-of-the-art laboratory filled with cutting-edge technology and innovative equipment.",
    "A girl with a teacher's glasses imparting valuable knowledge to her attentive stuffed animal students during a pretend classroom lesson on history, science, and literature.",
    "A boy with a wizard's hat casting powerful spells from an ancient book of magic in a mystical forest filled with towering trees, enchanted creatures, and hidden dangers.",
    "A child with a chef's hat proudly presenting a delicious assortment of freshly baked cookies, cakes, and pastries from the kitchen oven to share with friends and family.",
    "A kid with a doctor's coat tenderly caring for a sick stuffed animal patient with a gentle touch, a warm smile, and expert medical care to help them feel better.",
    "A child with a firefighter hat bravely rescuing a frightened kitten from the top of a tall tree with a sturdy ladder, a strong grip, and quick thinking to save the day.",
    "A girl with a painter's smock joyfully painting a vibrant mural on a city wall with bold strokes, bright colors, and creative imagination to express herself artistically.",
    "A boy with a police hat patrolling the neighborhood streets with a toy police car, a loud siren, and a watchful eye to maintain law and order and keep everyone safe.",
    "A child with a cowboy hat herding make-believe cattle on a pretend ranch in the wide-open plains of the wild west, riding horses, and roping cattle like a real cowboy.",
    "A kid with a superhero cape fearlessly defending the city from evildoers with superhuman strength, lightning-fast reflexes, and a strong sense of justice and righteousness.",
    "A child with a pirate hat embarking on an epic journey across the high seas to discover hidden treasure on a remote tropical island, facing danger, adventure, and excitement at every turn.",
    "A girl with a princess tiara attending a glamorous royal ball with her favorite stuffed animals as honored guests, dancing, laughing, and making cherished memories together.",
    "A boy with a knight's helmet gallantly rescuing a fair maiden from the clutches of an evil dragon in a faraway kingdom, wielding a mighty sword and showing bravery and chivalry in the face of danger.",
    "A child with a scuba mask exploring the wonders of an underwater world teeming with colorful fish, coral reefs, and exotic sea creatures, swimming, diving, and discovering new and amazing sights beneath the waves.",
    "A kid with a safari hat photographing majestic lions, elephants, and giraffes on an exciting safari adventure in the African savanna, capturing breathtaking images of wildlife in their natural habitat.",
    "A child with a scientist's goggles conducting groundbreaking experiments in a state-of-the-art laboratory filled with cutting-edge technology and innovative equipment, pushing the boundaries of knowledge and discovery in the pursuit of scientific excellence.",
    "A girl with a teacher's glasses imparting valuable knowledge to her attentive stuffed animal students during a pretend classroom lesson on history, science, and literature, inspiring a love of learning and curiosity about the world around them.",
    "A boy with a wizard's hat casting powerful spells from an ancient book of magic in a mystical forest filled with towering trees, enchanted creatures, and hidden dangers, harnessing the forces of nature and the power of the arcane to overcome obstacles and defeat enemies.",
    "A child with a chef's hat proudly presenting a delicious assortment of freshly baked cookies, cakes, and pastries from the kitchen oven to share with friends and family, spreading joy and happiness with the simple pleasure of homemade treats made with love and care.",
    "A kid with a doctor's coat tenderly caring for a sick stuffed animal patient with a gentle touch, a warm smile, and expert medical care to help them feel better, demonstrating compassion, empathy, and kindness towards those in need.",
    "A child with a firefighter hat bravely rescuing a frightened kitten from the top of a tall tree with a sturdy ladder, a strong grip, and quick thinking to save the day, embodying the spirit of heroism and selflessness in the face of danger.",
    "A girl with a painter's smock joyfully painting a vibrant mural on a city wall with bold strokes, bright colors, and creative imagination to express herself artistically, turning a blank canvas into a work of art that inspires and delights all who see it.",
    "A boy with a police hat patrolling the neighborhood streets with a toy police car, a loud siren, and a watchful eye to maintain law and order and keep everyone safe, serving and protecting the community with courage, dedication, and integrity.",
    "A child with a cowboy hat herding make-believe cattle on a pretend ranch in the wide-open plains of the wild west, riding horses, and roping cattle like a real cowboy, embracing the spirit of adventure and independence on the rugged frontier.",
    "A kid with a superhero cape fearlessly defending the city from evildoers with superhuman strength, lightning-fast reflexes, and a strong sense of justice and righteousness, fighting for truth, justice, and the greater good with unwavering determination and unwavering resolve.",
    "A child with a pirate hat embarking on an epic journey across the high seas to discover hidden treasure on a remote tropical island, facing danger, adventure, and excitement at every turn, navigating treacherous waters and braving the elements to find fortune and glory.",
    "A girl with a princess tiara attending a glamorous royal ball with her favorite stuffed animals as honored guests, dancing, laughing, and making cherished memories together, reveling in the magic and wonder of a fairy tale come to life in a grand and majestic palace.",
    "A boy with a knight's helmet gallantly rescuing a fair maiden from the clutches of an evil dragon in a faraway kingdom, wielding a mighty sword and showing bravery and chivalry in the face of danger, proving himself to be a true hero and champion of the realm.",
    "A child with a scuba mask exploring the wonders of an underwater world teeming with colorful fish, coral reefs, and exotic sea creatures, swimming, diving, and discovering new and amazing sights beneath the waves, embarking on an aquatic adventure like no other.",
    "A kid with a safari hat photographing majestic lions, elephants, and giraffes on an exciting safari adventure in the African savanna, capturing breathtaking images of wildlife in their natural habitat, documenting the beauty and splendor of the animal kingdom for future generations to enjoy.",
    "A child with a scientist's goggles conducting groundbreaking experiments in a state-of-the-art laboratory filled with cutting-edge technology and innovative equipment, pushing the boundaries of knowledge and discovery in the pursuit of scientific excellence, unlocking the secrets of the universe one discovery at a time.",
    "A girl with a teacher's glasses imparting valuable knowledge to her attentive stuffed animal students during a pretend classroom lesson on history, science, and literature, inspiring a love of learning and curiosity about the world around them, shaping the minds of tomorrow's leaders with wisdom and compassion.",
    "A boy with a wizard's hat casting powerful spells from an ancient book of magic in a mystical forest filled with towering trees, enchanted creatures, and hidden dangers, harnessing the forces of nature and the power of the arcane to overcome obstacles and defeat enemies, embarking on a magical adventure of epic proportions.",
    "A child with a chef's hat proudly presenting a delicious assortment of freshly baked cookies, cakes, and pastries from the kitchen oven to share with friends and family, spreading joy and happiness with the simple pleasure of homemade treats made with love and care, creating memories that last a lifetime.",
    "A kid with a doctor's coat tenderly caring for a sick stuffed animal patient with a gentle touch, a warm smile, and expert medical care to help them feel better, demonstrating compassion, empathy, and kindness towards those in need, embodying the healing power of love and empathy in action.",
    "A child with a firefighter hat bravely rescuing a frightened kitten from the top of a tall tree with a sturdy ladder, a strong grip, and quick thinking to save the day, embodying the spirit of heroism and selflessness in the face of danger, inspiring others with acts of courage and bravery.",
    "A girl with a painter's smock joyfully painting a vibrant mural on a city wall with bold strokes, bright colors, and creative imagination to express herself artistically, turning a blank canvas into a work of art that inspires and delights all who see it, leaving a lasting legacy of beauty and creativity for future generations to enjoy.",
    "A boy with a police hat patrolling the neighborhood streets with a toy police car, a loud siren, and a watchful eye to maintain law and order and keep everyone safe, serving and protecting the community with courage, dedication, and integrity, upholding the values of justice and equality for all.",
    "A child with a cowboy hat herding make-believe cattle on a pretend ranch in the wide-open plains of the wild west, riding horses, and roping cattle like a real cowboy, embracing the spirit of adventure and independence on the rugged frontier, living life with courage, determination, and a sense of adventure.",
    "A kid with a superhero cape fearlessly defending the city from evildoers with superhuman strength, lightning-fast reflexes, and a strong sense of justice and righteousness, fighting for truth, justice, and the greater good with unwavering determination and unwavering resolve, inspiring hope and courage in the hearts of all who believe.",
    "A child with a pirate hat embarking on an epic journey across the high seas to discover hidden treasure on a remote tropical island, facing danger, adventure, and excitement at every turn, navigating treacherous waters and braving the elements to find fortune and glory, embracing the pirate's life with courage, daring, and a spirit of adventure."
]
os.makedirs("child", exist_ok=True)
for prompt in prompts:
    res = t2i_pipe(prompt)[0]
    res.save(f"child/{prompt}.png")

# i = 0
# #repeat res for the same prompt for 100 times, for each one save it with unique name as png in the folder dogs
# os.makedirs("dogs", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a dog")[0]
#     res.save(f"dogs/dog_{i}.png")
#     i+=1
# i=0
# os.makedirs("man", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a man")[0]
#     res.save(f"man/man_{i}.png")
#     i += 1
# i=0
# os.makedirs("kid", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a kid")[0]
#     res.save(f"kid/kid_{i}.png")
#     i += 1
# i=0
# os.makedirs("nature", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a nature")[0]
#     res.save(f"nature/nature_{i}.png")
#     i += 1
#
# i=0
# os.makedirs("building", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a building")[0]
#     res.save(f"building/building_{i}.png")
#     i += 1
#
# i=0
# os.makedirs("car", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a car")[0]
#     res.save(f"car/car_{i}.png")
#     i += 1
#
# i=0
# os.makedirs("cat", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a cat")[0]
#     res.save(f"cat/cat_{i}.png")
#     i += 1
# i=0
# os.makedirs("tree", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a tree")[0]
#     res.save(f"tree/tree_{i}.png")
#     i += 1
# i=0
# os.makedirs("flower", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a flower")[0]
#     res.save(f"flower/flower_{i}.png")
#     i += 1
# i=0
# os.makedirs("fruit", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a fruit")[0]
#     res.save(f"fruit/fruit_{i}.png")
#     i += 1
# i=0
# os.makedirs("vegetable", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a vegetable")[0]
#     res.save(f"vegetable/vegetable_{i}.png")
#     i += 1
# i=0
# os.makedirs("sky", exist_ok=True)
#
# for i in range(100):
#     res = t2i_pipe("A photo of a sky")[0]
#     res.save(f"sky/sky_{i}.png")
#     i += 1
# i=0
# os.makedirs("cloud", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a cloud")[0]
#     res.save(f"cloud/cloud_{i}.png")
#     i += 1
# i=0
# os.makedirs("mountain", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a mountain")[0]
#     res.save(f"mountain/mountain_{i}.png")
#     i += 1
# i=0
# os.makedirs("river", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a river")[0]
#     res.save(f"river/river_{i}.png")
#     i += 1
# i=0
# os.makedirs("sea", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a sea")[0]
#     res.save(f"sea/sea_{i}.png")
#     i += 1
# i=0
# os.makedirs("beach", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a beach")[0]
#     res.save(f"beach/beach_{i}.png")
#     i += 1
# i=0
# os.makedirs("desert", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a desert")[0]
#     res.save(f"desert/desert_{i}.png")
#     i += 1
# i=0
# os.makedirs("forest", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a forest")[0]
#     res.save(f"forest/forest_{i}.png")
#     i += 1
# i=0
# os.makedirs("ball", exist_ok=True)
# for i in range(100):
#     res = t2i_pipe("A photo of a ball")[0]
#     res.save(f"ball/ball_{i}.png")
#     i += 1