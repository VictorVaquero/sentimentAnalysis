import random, string, datetime

import request



metodoHTTP = "GET"
cabecera = "https://api.twitter.com/1.1/search/tweets.json?"


# Clave unica de mi cuenta twitter
consumer_key = {"oauth_consumer_key": "cO3OOhllb2NWHMm70fGpVeFai"}

# Clave alfanumerica unica para cada peticion
x = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
nonce = {"oauth_nonce": x}

# Metodo autorizado twitter
signature_method = {"oauth_signature_method": "HMAC-SHA1"}

# Tiempo ntp
diff = datetime.datetime.utcnow() - datetime.datetime(1900, 1, 1, 0, 0, 0)
tiempo = {"oauth_timestamp":timedelta.total_seconds(diff)}

# Token de aceso, unico de mi cuenta
token = {"oauth_token": "2186671160-6fohfZTgQOi1GTL2YQdc5C8k9f3wceZCLYddZNa"}

# Version
version = {"oauth_version":"1.0"}




