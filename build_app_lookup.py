"""
Build an optimized app-name lookup dictionary from the raw android_apps_names.json.

Problems in the raw data this script fixes:
──────────────────────────────────────────
1. 50k entries → ~90% are niche/obscure → drops them to keep only high-value apps.
2. "fancyname" includes long subtitles ("Messenger — Text and Video Chat for Free")
   → strips everything after " - ", ": ", " — ", " | " to get the *display name*.
3. No aliases → adds OCR-friendly aliases (e.g. "IG" → Instagram, "YT" → YouTube).
4. Generic names like "Calculator", "Gallery" exist multiple times → keeps ONE canonical.
5. Encoding issues (1711 entries with replacement chars) → removes them.
6. Duplicate fancynames across different packages → deduplicates.
7. basename field is mostly useless (293× "android", 267× "chromium") → discarded.

Output: app_lookup.json — a compact dict optimized for rapidfuzz matching.
"""

import json
import re
from collections import defaultdict

# ── Load raw data ──────────────────────────────────────────────────────
with open('android_apps_names.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

print(f"Raw entries: {len(raw)}")

# ── Step 1: Filter out garbage ─────────────────────────────────────────
valid = []
for entry in raw:
    fn = entry.get('fancyname')
    if fn is None:
        continue
    # Remove encoding-broken entries
    if '\ufffd' in fn or '�' in fn:
        continue
    # Remove too-short names (1-2 chars → false positives galore)
    if len(fn.strip()) <= 2:
        continue
    valid.append(entry)

print(f"After cleaning None/encoding/short: {len(valid)}")

# ── Step 2: Extract clean display name from fancyname ──────────────────
SEPARATOR_RE = re.compile(r'\s*(?:\s[-–—]\s|:\s|\s\|\s)')

def clean_name(fancyname: str) -> str:
    """Strip subtitle/tagline, normalize whitespace."""
    # Take only the part before the first separator
    name = SEPARATOR_RE.split(fancyname)[0]
    name = name.strip()
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name)
    return name

# ── Step 3: Deduplicate by clean name ──────────────────────────────────
# Keep the first occurrence (raw data is roughly popularity-ordered)
seen_names = {}  # lowercase_clean_name → entry
for entry in valid:
    clean = clean_name(entry['fancyname'])
    key = clean.lower()
    if key not in seen_names:
        seen_names[key] = {
            'name': clean,
            'full_name': entry['fancyname'].strip(),
            'package': entry['appname'],
        }

print(f"After dedup by clean name: {len(seen_names)}")

# ── Step 4: Remove generic/ambiguous single-word names ─────────────────
# These cause massive false positives when fuzzy-matched against OCR text
GENERIC_BLOCKLIST = {
    # System UI words
    'settings', 'clock', 'calculator', 'calendar', 'camera', 'phone',
    'contacts', 'messages', 'music', 'gallery', 'files', 'notes',
    'weather', 'browser', 'video', 'audio', 'radio', 'recorder',
    'maps', 'store', 'shop', 'mail', 'email', 'chat', 'news',
    'photo', 'editor', 'scanner', 'cleaner', 'launcher', 'keyboard',
    'timer', 'compass', 'alarm', 'dialer', 'wallpaper', 'memo',
    'backup', 'update', 'search', 'home', 'theme', 'themes',
    # Common UI words that aren't apps
    'ok', 'yes', 'no', 'cancel', 'done', 'next', 'back', 'open',
    'close', 'send', 'share', 'save', 'delete', 'edit', 'add',
    'new', 'more', 'menu', 'info', 'help', 'about', 'login',
    'sign in', 'sign up', 'loading', 'error', 'warning', 'success',
    # Overly common English words
    'free', 'game', 'games', 'live', 'play', 'pro', 'lite', 'go',
    'app', 'apps', 'tv', 'fm', 'hd', 'vpn', 'pdf', 'hub',
    'smart', 'super', 'fast', 'easy', 'cool', 'best', 'top',
    'my', 'your', 'the', 'it', 'me', 'we', 'us',
    # Common game titles that are just words
    'solitaire', 'sudoku', 'chess', 'checkers', 'poker', 'puzzle',
    'racing', 'shooting', 'runner', 'block', 'blocks', 'ball',
    'bubble', 'word', 'quiz', '2048', 'snake', 'tetris',
    # No info / placeholder
    'no info', 'test', 'demo', 'debug', 'sample', 'example',
}

filtered = {}
for key, entry in seen_names.items():
    if key in GENERIC_BLOCKLIST:
        continue
    # Also skip if it's a single very short word (<=4 chars) that's not a well-known brand
    if len(key) <= 4 and ' ' not in key:
        # Keep only if it's a known brand
        KNOWN_SHORT = {'uber', 'lyft', 'zoom', 'bolt', 'wish', 'temu', 'shein',
                       'asos', 'ikea', 'imdb', 'yelp', 'etsy', 'zara', 'tubi',
                       'roku', 'hulu', 'kik', 'line', 'viber', 'wire'}  # will add more
        if key not in KNOWN_SHORT:
            continue
    filtered[key] = entry

print(f"After removing generic/short: {len(filtered)}")

# ── Step 5: Build the final lookup with aliases ────────────────────────
# Structure: list of {"name": str, "aliases": [str], "package": str}

# Hard-coded aliases for top apps that OCR commonly encounters
# These are abbreviations, common misspellings, and alternative names
ALIAS_MAP = {
    # Social Media
    'instagram':        ['IG', 'Insta', 'lnstagram'],
    'facebook':         ['FB', 'fb', 'Faceb00k'],
    'twitter':          ['X', 'X (Twitter)'],
    'x':                ['Twitter', 'X (formerly Twitter)'],
    'whatsapp':         ['WA', 'Whats App', 'What\'s App', 'WhatsUp', 'Watsapp', 'Whats app'],
    'whatsapp messenger': ['WA', 'Whats App', 'What\'s App', 'WhatsUp', 'Watsapp'],
    'snapchat':         ['Snap', 'SC'],
    'tiktok':           ['Tik Tok', 'TT', 'Tik-Tok'],
    'telegram':         ['TG', 'Tele gram'],
    'discord':          ['DC'],
    'linkedin':         ['Linked In', 'Linkedln'],
    'pinterest':        ['Pin', 'Pinterst'],
    'reddit':           ['Redd it'],
    'tumblr':           ['Tumbler'],
    'wechat':           ['WeChat', 'We Chat', '微信'],
    'threads':          ['Threads by Instagram'],

    # Messaging
    'messenger':        ['FB Messenger', 'Facebook Messenger', 'Msg'],
    'signal':           ['Signal Private Messenger'],
    'viber':            ['Viber Messenger'],

    # Video/Streaming
    'youtube':          ['YT', 'You Tube', 'You tube', 'YouTub'],
    'youtube music':    ['YT Music', 'YouTube Music'],
    'netflix':          ['Netfllx', 'Netfilx', 'NETFLIX'],
    'spotify':          ['Spotfy', 'Spotlfy', 'SPOTIFY'],
    'disney+':          ['Disney Plus', 'DisneyPlus', 'Disney +'],
    'hbo max':          ['HBO', 'HBOMax', 'Max'],
    'amazon prime video': ['Prime Video', 'Prime', 'Amazon Video'],
    'apple music':      ['Apple Music'],
    'deezer':           ['Deezer'],
    'soundcloud':       ['Sound Cloud', 'SoundCIoud'],
    'twitch':           ['Twltch'],
    'hulu':             ['HULU'],

    # Google Apps
    'google chrome':    ['Chrome', 'Chrom', 'Googl Chrome'],
    'google maps':      ['Maps', 'GMaps', 'Google Map', 'Gmap'],
    'gmail':            ['G Mail', 'G-Mail', 'Gmai', 'Gmall'],
    'google drive':     ['Drive', 'GDrive', 'G Drive'],
    'google photos':    ['Photos', 'G Photos'],
    'google play':      ['Play Store', 'Google Play Store', 'Play store'],
    'google translate': ['Translate', 'Google Translator', 'G Translate'],
    'google calendar':  ['Calendar', 'G Calendar', 'GCal'],
    'google meet':      ['Meet', 'G Meet', 'Gmeet'],
    'google classroom': ['Classroom'],
    'google authenticator': ['Authenticator', 'G Auth'],
    'google sheets':    ['Sheets', 'G Sheets'],
    'google docs':      ['Docs', 'G Docs'],
    'google slides':    ['Slides', 'G Slides'],
    'google home':      ['Home', 'G Home'],
    'google earth':     ['Earth'],
    'google pay':       ['GPay', 'G Pay'],
    'google lens':      ['Lens', 'G Lens'],
    'google keep':      ['Keep', 'G Keep'],

    # Microsoft
    'microsoft outlook':['Outlook', 'MS Outlook'],
    'microsoft teams':  ['Teams', 'MS Teams'],
    'microsoft word':   ['Word', 'MS Word'],
    'microsoft excel':  ['Excel', 'MS Excel'],
    'microsoft powerpoint': ['PowerPoint', 'PPT', 'MS PPT'],
    'microsoft onenote':['OneNote', 'One Note'],
    'microsoft onedrive': ['OneDrive', 'One Drive'],
    'microsoft edge':   ['Edge', 'MS Edge'],
    'microsoft to do':  ['To Do', 'MS To Do'],
    'skype':            ['Skype'],

    # Ride/Transport
    'uber':             ['UBER', 'Uberr'],
    'uber eats':        ['UberEats', 'Uber Eat', 'UberEat'],
    'lyft':             ['LYFT', 'Lft'],
    'bolt':             ['BOLT', 'Taxify'],
    'grab':             ['GrabTaxi', 'GRAB'],
    'waze':             ['WAZE', 'Waz'],

    # Shopping
    'amazon':           ['Amazon Shopping', 'AMAZON'],
    'amazon shopping':  ['Amazon'],
    'ebay':             ['eBay', 'EBAY', 'e-Bay', 'e bay'],
    'aliexpress':       ['Ali Express', 'ALIEXPRESS', 'AliExpres'],
    'wish':             ['WISH'],
    'shein':            ['SHEIN', 'SheIn', 'She in'],
    'temu':             ['TEMU'],
    'zalando':          ['ZALANDO'],
    'etsy':             ['ETSY'],

    # Food Delivery
    'doordash':         ['Door Dash', 'DoorDash'],
    'grubhub':          ['Grub Hub', 'GrubHub'],
    'deliveroo':        ['Deliveroo'],
    'just eat':         ['JustEat', 'Just Eat'],

    # Banking/Payment
    'paypal':           ['Pay Pal', 'PayPaI', 'PAYPAL'],
    'venmo':            ['VENMO'],
    'cash app':         ['CashApp', 'Cash App', '$Cash'],
    'revolut':          ['REVOLUT'],
    'n26':              ['N26'],

    # Productivity
    'notion':           ['Notion'],
    'slack':            ['SLACK', 'SIack'],
    'trello':           ['TRELLO'],
    'evernote':         ['Evernote', 'Ever Note'],
    'todoist':          ['Todoist'],
    'dropbox':          ['Drop Box', 'DropBox', 'DROPBOX'],

    # Camera/Photo
    'google camera':    ['Camera', 'GCam'],

    # Health
    'fitbit':           ['FitBit', 'Fit Bit', 'FITBIT'],
    'myfitnesspal':     ['MyFitnessPal', 'My Fitness Pal', 'MFP'],
    'strava':           ['STRAVA'],
    'samsung health':   ['S Health', 'Samsung Health'],
    'apple health':     ['Health'],

    # Travel
    'airbnb':           ['AirBnB', 'Air BnB', 'AIRBNB'],
    'booking.com':      ['Booking', 'BOOKING', 'Booking com'],
    'tripadvisor':      ['Trip Advisor', 'TripAdvisor', 'Trip advisor'],
    'google flights':   ['Flights'],
    'expedia':          ['EXPEDIA'],

    # News/Read
    'kindle':           ['Amazon Kindle', 'KINDLE'],
    'pocket':           ['GetPocket', 'Get Pocket'],
    'flipboard':        ['Flip Board', 'FlipBoard'],

    # Utility
    'shazam':           ['SHAZAM', 'Shazzam'],
    'calculator+':      ['Calculator Plus'],
    'adobe acrobat reader': ['Acrobat', 'Adobe Reader', 'PDF Reader'],
    'calm':             ['CALM'],
    'duolingo':         ['Duo Lingo', 'DuoLingo', 'DUOLINGO'],
    'google clock':     ['Clock'],
    'google calculator':['Calculator'],
}

# Build the final lookup list
lookup_entries = []
for key, entry in filtered.items():
    aliases = ALIAS_MAP.get(key, [])
    lookup_entries.append({
        'name': entry['name'],
        'aliases': aliases,
        'package': entry['package'],
    })

# ── Step 6: Add high-value apps that MUST be present ───────────────────
# (in case they were filtered out or missing from the raw data)
MUST_HAVE_APPS = [
    {'name': 'TikTok',     'aliases': ['Tik Tok', 'TT', 'Tik-Tok'], 'package': 'com.zhiliaoapp.musically'},
    {'name': 'ChatGPT',    'aliases': ['Chat GPT', 'Open AI', 'OpenAI Chat'], 'package': 'com.openai.chatgpt'},
    {'name': 'Gemini',     'aliases': ['Google Gemini', 'Bard'], 'package': 'com.google.android.apps.bard'},
    {'name': 'Copilot',    'aliases': ['Microsoft Copilot', 'MS Copilot'], 'package': 'com.microsoft.copilot'},
    {'name': 'Claude',     'aliases': ['Anthropic Claude'], 'package': 'com.anthropic.claude'},
    {'name': 'Threads',    'aliases': ['Threads by Instagram'], 'package': 'com.instagram.barcelona'},
    {'name': 'BeReal',     'aliases': ['Be Real', 'BeReal.'], 'package': 'com.bereal.ft'},
    {'name': 'Lemon8',     'aliases': ['Lemon 8'], 'package': 'com.bd.lemon8'},
    {'name': 'Apple Music', 'aliases': [], 'package': 'com.apple.android.music'},
    {'name': 'Disney+',    'aliases': ['Disney Plus', 'DisneyPlus', 'Disney +'], 'package': 'com.disney.disneyplus'},
    {'name': 'Max',        'aliases': ['HBO Max', 'HBO', 'HBOMax'], 'package': 'com.wbd.stream'},
    {'name': 'Paramount+', 'aliases': ['Paramount Plus'], 'package': 'com.cbs.app'},
    {'name': 'Peacock',    'aliases': ['Peacock TV', 'NBC Peacock'], 'package': 'com.peacocktv.peacockandroid'},
    {'name': 'Tubi',       'aliases': ['Tubi TV'], 'package': 'com.tubitv'},
    {'name': 'Crunchyroll', 'aliases': ['Crunchy Roll', 'Crunchyrol'], 'package': 'com.crunchyroll.crunchyroid'},
    {'name': 'Notion',     'aliases': [], 'package': 'notion.id'},
    {'name': 'Canva',      'aliases': ['CANVA'], 'package': 'com.canva.editor'},
    {'name': 'Figma',      'aliases': ['FIGMA'], 'package': 'com.figma.mirror'},
    {'name': 'Slack',      'aliases': ['SLACK', 'SIack'], 'package': 'com.Slack'},
    {'name': 'Zoom',       'aliases': ['Zoom Meetings', 'ZOOM', 'Zoom Meeting'], 'package': 'us.zoom.videomeetings'},
    {'name': 'Microsoft Teams', 'aliases': ['Teams', 'MS Teams'], 'package': 'com.microsoft.teams'},
    {'name': 'Google Meet', 'aliases': ['Meet', 'G Meet', 'Gmeet'], 'package': 'com.google.android.apps.tachyon'},
    {'name': 'FaceTime',   'aliases': ['Face Time'], 'package': 'com.apple.facetime'},
    {'name': 'Telegram',   'aliases': ['TG', 'Tele gram'], 'package': 'org.telegram.messenger'},
    {'name': 'Signal',     'aliases': ['Signal Private Messenger'], 'package': 'org.thoughtcrime.securesms'},
    {'name': 'WeChat',     'aliases': ['We Chat', '微信'], 'package': 'com.tencent.mm'},
    {'name': 'LINE',       'aliases': ['Line'], 'package': 'jp.naver.line.android'},
    {'name': 'KakaoTalk',  'aliases': ['Kakao Talk', 'Kakao', '카카오톡'], 'package': 'com.kakao.talk'},
    {'name': 'Tinder',     'aliases': ['TINDER'], 'package': 'com.tinder'},
    {'name': 'Bumble',     'aliases': ['BUMBLE'], 'package': 'com.bumble.app'},
    {'name': 'Hinge',      'aliases': ['HINGE'], 'package': 'co.hinge.app'},
    {'name': 'Cash App',   'aliases': ['CashApp', '$Cash'], 'package': 'com.squareup.cash'},
    {'name': 'Venmo',      'aliases': ['VENMO'], 'package': 'com.venmo'},
    {'name': 'Zelle',      'aliases': ['ZELLE'], 'package': 'com.zellepay.zelle'},
    {'name': 'Revolut',    'aliases': ['REVOLUT'], 'package': 'com.revolut.revolut'},
    {'name': 'Wise',       'aliases': ['TransferWise', 'Transfer Wise'], 'package': 'com.transferwise.android'},
    {'name': 'Robinhood',  'aliases': ['Robin Hood', 'Robinhood Markets'], 'package': 'com.robinhood.android'},
    {'name': 'Coinbase',   'aliases': ['Coin Base', 'CoinBase'], 'package': 'com.coinbase.android'},
    {'name': 'DoorDash',   'aliases': ['Door Dash', 'DOORDASH'], 'package': 'com.dd.doordash'},
    {'name': 'Grubhub',    'aliases': ['Grub Hub', 'GRUBHUB'], 'package': 'com.grubhub.android'},
    {'name': 'Instacart',  'aliases': ['Insta Cart', 'InstaCart'], 'package': 'com.instacart.client'},
    {'name': 'Postmates',  'aliases': ['Post Mates'], 'package': 'com.postmates.android'},
    {'name': 'Wolt',       'aliases': ['WOLT'], 'package': 'com.wolt.android'},
    {'name': 'Glovo',      'aliases': ['GLOVO'], 'package': 'com.glovo'},
    {'name': 'Rappi',      'aliases': ['RAPPI'], 'package': 'com.grability.rappi'},
    {'name': 'iFood',      'aliases': ['ifood', 'IFood'], 'package': 'br.com.brainweb.ifood'},
    {'name': 'Spotify',    'aliases': ['Spotfy', 'Spotlfy', 'SPOTIFY'], 'package': 'com.spotify.music'},
    {'name': 'Strava',     'aliases': ['STRAVA'], 'package': 'com.strava'},
    {'name': 'Nike Run Club', 'aliases': ['NRC', 'Nike Running'], 'package': 'com.nike.plusgps'},
    {'name': 'MyFitnessPal', 'aliases': ['My Fitness Pal', 'MFP'], 'package': 'com.myfitnesspal.android'},
    {'name': 'Duolingo',   'aliases': ['Duo Lingo', 'DuoLingo', 'DUOLINGO'], 'package': 'com.duolingo'},
    {'name': 'Khan Academy', 'aliases': ['KhanAcademy', 'Khan'], 'package': 'org.khanacademy.android'},
    {'name': 'Coursera',   'aliases': ['COURSERA'], 'package': 'org.coursera.android'},
    {'name': 'Photomath',  'aliases': ['Photo Math'], 'package': 'com.microblink.photomath'},
    {'name': 'Google Wallet', 'aliases': ['Wallet', 'G Wallet', 'Google Pay'], 'package': 'com.google.android.apps.walletnfcrel'},
    {'name': 'Apple Wallet', 'aliases': ['Wallet'], 'package': 'com.apple.wallet'},
    {'name': 'Samsung Pay', 'aliases': ['S Pay', 'Samsung Pay'], 'package': 'com.samsung.android.spay'},
    {'name': 'Samsung Internet', 'aliases': ['Samsung Browser', 'S Browser'], 'package': 'com.sec.android.app.sbrowser'},
    {'name': 'Samsung Notes', 'aliases': ['S Notes'], 'package': 'com.samsung.android.app.notes'},
    {'name': 'Samsung Gallery', 'aliases': ['S Gallery'], 'package': 'com.sec.android.gallery3d'},
    {'name': 'Apple Maps',  'aliases': [], 'package': 'com.apple.Maps'},
    {'name': 'iMessage',   'aliases': ['iMessages', 'i Message'], 'package': 'com.apple.MobileSMS'},
    {'name': 'Safari',     'aliases': [], 'package': 'com.apple.mobilesafari'},
    {'name': 'Firefox',    'aliases': ['Mozilla Firefox', 'Fire Fox'], 'package': 'org.mozilla.firefox'},
    {'name': 'Opera',      'aliases': ['Opera Browser', 'Opera Mini'], 'package': 'com.opera.browser'},
    {'name': 'Brave',      'aliases': ['Brave Browser'], 'package': 'com.brave.browser'},
    {'name': 'DuckDuckGo', 'aliases': ['Duck Duck Go', 'DDG'], 'package': 'com.duckduckgo.mobile.android'},
    {'name': 'Tor Browser', 'aliases': ['Tor'], 'package': 'org.torproject.torbrowser'},
    {'name': 'Shazam',     'aliases': ['SHAZAM', 'Shazzam'], 'package': 'com.shazam.android'},
    {'name': 'Adobe Acrobat', 'aliases': ['Acrobat', 'Adobe Reader', 'PDF Reader', 'Adobe Acrobat Reader'], 'package': 'com.adobe.reader'},
    {'name': 'VLC',        'aliases': ['VLC Player', 'VLC Media Player'], 'package': 'org.videolan.vlc'},
    {'name': 'Plex',       'aliases': ['PLEX', 'Plex Media'], 'package': 'com.plexapp.android'},
    {'name': 'Audible',    'aliases': ['Amazon Audible', 'AUDIBLE'], 'package': 'com.audible.application'},
    {'name': 'Podcasts',   'aliases': ['Google Podcasts', 'Apple Podcasts'], 'package': 'com.google.android.apps.podcasts'},
    {'name': 'Waze',       'aliases': ['WAZE'], 'package': 'com.waze'},
    {'name': 'Citymapper', 'aliases': ['City Mapper'], 'package': 'com.citymapper.app.release'},
    {'name': 'Moovit',     'aliases': ['MOOVIT'], 'package': 'com.tranzmate'},
    {'name': 'ProtonMail', 'aliases': ['Proton Mail', 'ProtonMail', 'Proton'], 'package': 'ch.protonmail.android'},
    {'name': 'Bitwarden',  'aliases': ['Bit Warden'], 'package': 'com.x8bit.bitwarden'},
    {'name': '1Password',  'aliases': ['1 Password', 'OnePassword'], 'package': 'com.agilebits.onepassword'},
    {'name': 'NordVPN',    'aliases': ['Nord VPN', 'NordVpn'], 'package': 'com.nordvpn.android'},
    {'name': 'ExpressVPN', 'aliases': ['Express VPN'], 'package': 'com.expressvpn.vpn'},
    {'name': 'Depop',      'aliases': ['DEPOP'], 'package': 'com.depop'},
    {'name': 'Vinted',     'aliases': ['VINTED'], 'package': 'com.vinted.android'},
    {'name': 'Mercari',    'aliases': ['MERCARI'], 'package': 'com.kouzoh.mercari'},
    {'name': 'Poshmark',   'aliases': ['Posh Mark'], 'package': 'com.poshmark.app'},
    {'name': 'IKEA',       'aliases': ['Ikea'], 'package': 'com.ingka.ikea.app'},
    {'name': 'Walgreens',  'aliases': [], 'package': 'com.walgreens.wag'},
    {'name': 'CVS',        'aliases': ['CVS Pharmacy'], 'package': 'com.cvs.launchers.cvs'},
    {'name': 'Target',     'aliases': [], 'package': 'com.target.ui'},
    {'name': 'Walmart',    'aliases': ['Wal Mart', 'Wal-Mart'], 'package': 'com.walmart.android'},
    {'name': 'Costco',     'aliases': ['COSTCO'], 'package': 'com.costco.app.android'},
]

# Add must-have apps, avoiding duplicates
existing_names = {e['name'].lower() for e in lookup_entries}
for app in MUST_HAVE_APPS:
    if app['name'].lower() not in existing_names:
        lookup_entries.append(app)
        existing_names.add(app['name'].lower())

print(f"Final lookup entries: {len(lookup_entries)}")

# ── Step 7: Build the search index ─────────────────────────────────────
# For rapidfuzz, we need a flat list of (search_term → canonical_name) mappings.
# This is the key data structure for efficient matching.

search_index = {}  # search_term_lower → {"name": canonical, "package": pkg}

for entry in lookup_entries:
    canonical = entry['name']
    pkg = entry['package']
    record = {'name': canonical, 'package': pkg}
    
    # Add the canonical name
    search_index[canonical.lower()] = record
    
    # Add all aliases
    for alias in entry.get('aliases', []):
        alias_lower = alias.lower()
        if alias_lower not in search_index:
            search_index[alias_lower] = record

print(f"Total search terms (with aliases): {len(search_index)}")

# ── Save outputs ────────────────────────────────────────────────────────
# Save the structured lookup
with open('app_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(lookup_entries, f, ensure_ascii=False, indent=2)

# Save the flat search index (for direct rapidfuzz use)
with open('app_search_index.json', 'w', encoding='utf-8') as f:
    json.dump(search_index, f, ensure_ascii=False, indent=2)

print(f"\nSaved: app_lookup.json ({len(lookup_entries)} entries)")
print(f"Saved: app_search_index.json ({len(search_index)} search terms)")

# Size comparison
import os
raw_size = os.path.getsize('android_apps_names.json')
lookup_size = os.path.getsize('app_lookup.json')
index_size = os.path.getsize('app_search_index.json')
print(f"\nSize comparison:")
print(f"  Raw:          {raw_size:>12,} bytes ({raw_size/1024/1024:.1f} MB)")
print(f"  Lookup:       {lookup_size:>12,} bytes ({lookup_size/1024:.0f} KB)")
print(f"  Search index: {index_size:>12,} bytes ({index_size/1024:.0f} KB)")
