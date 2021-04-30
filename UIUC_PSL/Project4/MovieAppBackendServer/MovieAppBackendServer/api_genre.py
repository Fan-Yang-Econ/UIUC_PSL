"""
"""

import logging

GENRE_SUMMARY = [{'Genres': 'Action', 'Top5_most_rating_id': [260, 1196, 1210, 480, 2028], 'Top5_highest_rating_id': [2905, 2019, 858, 1198, 260],
                  'Top5_most_rating_name': ['Star Wars: Episode IV - A New Hope (1977)', 'Star Wars: Episode V - The Empire Strikes Back (1980)',
                                            'Star Wars: Episode VI - Return of the Jedi (1983)', 'Jurassic Park (1993)',
                                            'Saving Private Ryan (1998)'],
                  'Top5_highest_rating_name': ['Sanjuro (1962)', 'Seven Samurai (The Magnificent Seven) (Shichinin no samurai) (1954)',
                                               'Godfather, The (1972)', 'Raiders of the Lost Ark (1981)',
                                               'Star Wars: Episode IV - A New Hope (1977)']},
                 {'Genres': 'Adventure', 'Top5_most_rating_id': [260, 1196, 1210, 480, 1580], 'Top5_highest_rating_id': [3172, 2905, 1198, 260, 1204],
                  'Top5_most_rating_name': ['Star Wars: Episode IV - A New Hope (1977)', 'Star Wars: Episode V - The Empire Strikes Back (1980)',
                                            'Star Wars: Episode VI - Return of the Jedi (1983)', 'Jurassic Park (1993)', 'Men in Black (1997)'],
                  'Top5_highest_rating_name': ['Ulysses (Ulisse) (1954)', 'Sanjuro (1962)', 'Raiders of the Lost Ark (1981)',
                                               'Star Wars: Episode IV - A New Hope (1977)', 'Lawrence of Arabia (1962)']},
                 {'Genres': 'Animation', 'Top5_most_rating_id': [1, 2987, 2355, 3114, 588], 'Top5_highest_rating_id': [745, 1148, 720, 1223, 3429],
                  'Top5_most_rating_name': ['Toy Story (1995)', 'Who Framed Roger Rabbit? (1988)', "Bug's Life, A (1998)", 'Toy Story 2 (1999)',
                                            'Aladdin (1992)'], 'Top5_highest_rating_name': ['Close Shave, A (1995)', 'Wrong Trousers, The (1993)',
                                                                                            'Wallace & Gromit: The Best of Aardman Animation (1996)',
                                                                                            'Grand Day Out, A (1992)', 'Creature Comforts (1990)']},
                 {'Genres': "Children's", 'Top5_most_rating_id': [1097, 1, 34, 919, 2355], 'Top5_highest_rating_id': [919, 3114, 1, 2761, 1023],
                  'Top5_most_rating_name': ['E.T. the Extra-Terrestrial (1982)', 'Toy Story (1995)', 'Babe (1995)', 'Wizard of Oz, The (1939)',
                                            "Bug's Life, A (1998)"],
                  'Top5_highest_rating_name': ['Wizard of Oz, The (1939)', 'Toy Story 2 (1999)', 'Toy Story (1995)', 'Iron Giant, The (1999)',
                                               'Winnie the Pooh and the Blustery Day (1968)']},
                 {'Genres': 'Comedy', 'Top5_most_rating_id': [2858, 1270, 1580, 2396, 1197], 'Top5_highest_rating_id': [3233, 1830, 3607, 745, 1148],
                  'Top5_most_rating_name': ['American Beauty (1999)', 'Back to the Future (1985)', 'Men in Black (1997)',
                                            'Shakespeare in Love (1998)', 'Princess Bride, The (1987)'],
                  'Top5_highest_rating_name': ['Smashing Time (1967)', 'Follow the Bitch (1998)', 'One Little Indian (1973)', 'Close Shave, A (1995)',
                                               'Wrong Trousers, The (1993)']},
                 {'Genres': 'Crime', 'Top5_most_rating_id': [608, 1617, 858, 296, 50], 'Top5_highest_rating_id': [3656, 858, 50, 3517, 3435],
                  'Top5_most_rating_name': ['Fargo (1996)', 'L.A. Confidential (1997)', 'Godfather, The (1972)', 'Pulp Fiction (1994)',
                                            'Usual Suspects, The (1995)'],
                  'Top5_highest_rating_name': ['Lured (1947)', 'Godfather, The (1972)', 'Usual Suspects, The (1995)', 'Bells, The (1926)',
                                               'Double Indemnity (1944)']},
                 {'Genres': 'Documentary', 'Top5_most_rating_id': [2064, 246, 162, 3007, 1147],
                  'Top5_highest_rating_id': [3881, 787, 3338, 2930, 128],
                  'Top5_most_rating_name': ['Roger & Me (1989)', 'Hoop Dreams (1994)', 'Crumb (1994)', 'American Movie (1999)',
                                            'When We Were Kings (1996)'],
                  'Top5_highest_rating_name': ['Bittersweet Motel (2000)', 'Gate of Heavenly Peace, The (1995)', 'For All Mankind (1989)',
                                               'Return with Honor (1998)', "Jupiter's Wife (1994)"]},
                 {'Genres': 'Drama', 'Top5_most_rating_id': [2858, 1196, 2028, 593, 608], 'Top5_highest_rating_id': [3382, 989, 3607, 3245, 53],
                  'Top5_most_rating_name': ['American Beauty (1999)', 'Star Wars: Episode V - The Empire Strikes Back (1980)',
                                            'Saving Private Ryan (1998)', 'Silence of the Lambs, The (1991)', 'Fargo (1996)'],
                  'Top5_highest_rating_name': ['Song of Freedom (1936)', 'Schlafes Bruder (Brother of Sleep) (1995)', 'One Little Indian (1973)',
                                               'I Am Cuba (Soy Cuba/Ya Kuba) (1964)', 'Lamerica (1994)']},
                 {'Genres': 'Fantasy', 'Top5_most_rating_id': [260, 1097, 2628, 2174, 2797], 'Top5_highest_rating_id': [260, 792, 1097, 247, 1073],
                  'Top5_most_rating_name': ['Star Wars: Episode IV - A New Hope (1977)', 'E.T. the Extra-Terrestrial (1982)',
                                            'Star Wars: Episode I - The Phantom Menace (1999)', 'Beetlejuice (1988)', 'Big (1988)'],
                  'Top5_highest_rating_name': ['Star Wars: Episode IV - A New Hope (1977)', 'Hungarian Fairy Tale, A (1987)',
                                               'E.T. the Extra-Terrestrial (1982)', 'Heavenly Creatures (1994)',
                                               'Willy Wonka and the Chocolate Factory (1971)']},
                 {'Genres': 'Film-Noir', 'Top5_most_rating_id': [1617, 541, 2987, 1252, 913], 'Top5_highest_rating_id': [922, 3435, 913, 1252, 1267],
                  'Top5_most_rating_name': ['L.A. Confidential (1997)', 'Blade Runner (1982)', 'Who Framed Roger Rabbit? (1988)', 'Chinatown (1974)',
                                            'Maltese Falcon, The (1941)'],
                  'Top5_highest_rating_name': ['Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)', 'Double Indemnity (1944)',
                                               'Maltese Falcon, The (1941)', 'Chinatown (1974)', 'Manchurian Candidate, The (1962)']},
                 {'Genres': 'Horror', 'Top5_most_rating_id': [2716, 1214, 1387, 1219, 2710], 'Top5_highest_rating_id': [3280, 1278, 1219, 1214, 1258],
                  'Top5_most_rating_name': ['Ghostbusters (1984)', 'Alien (1979)', 'Jaws (1975)', 'Psycho (1960)', 'Blair Witch Project, The (1999)'],
                  'Top5_highest_rating_name': ['Baby, The (1973)', 'Young Frankenstein (1974)', 'Psycho (1960)', 'Alien (1979)',
                                               'Shining, The (1980)']},
                 {'Genres': 'Musical', 'Top5_most_rating_id': [919, 588, 1220, 2657, 364], 'Top5_highest_rating_id': [899, 919, 1288, 1066, 914],
                  'Top5_most_rating_name': ['Wizard of Oz, The (1939)', 'Aladdin (1992)', 'Blues Brothers, The (1980)',
                                            'Rocky Horror Picture Show, The (1975)', 'Lion King, The (1994)'],
                  'Top5_highest_rating_name': ["Singin' in the Rain (1952)", 'Wizard of Oz, The (1939)', 'This Is Spinal Tap (1984)',
                                               'Shall We Dance? (1937)', 'My Fair Lady (1964)']},
                 {'Genres': 'Mystery', 'Top5_most_rating_id': [1617, 924, 648, 3176, 1252], 'Top5_highest_rating_id': [578, 904, 1212, 913, 1252],
                  'Top5_most_rating_name': ['L.A. Confidential (1997)', '2001: A Space Odyssey (1968)', 'Mission: Impossible (1996)',
                                            'Talented Mr. Ripley, The (1999)', 'Chinatown (1974)'],
                  'Top5_highest_rating_name': ['Hour of the Pig, The (1993)', 'Rear Window (1954)', 'Third Man, The (1949)',
                                               'Maltese Falcon, The (1941)', 'Chinatown (1974)']},
                 {'Genres': 'Romance', 'Top5_most_rating_id': [1210, 2396, 1197, 1265, 356], 'Top5_highest_rating_id': [3888, 912, 3307, 1197, 898],
                  'Top5_most_rating_name': ['Star Wars: Episode VI - Return of the Jedi (1983)', 'Shakespeare in Love (1998)',
                                            'Princess Bride, The (1987)', 'Groundhog Day (1993)', 'Forrest Gump (1994)'],
                  'Top5_highest_rating_name': ['Skipped Parts (2000)', 'Casablanca (1942)', 'City Lights (1931)', 'Princess Bride, The (1987)',
                                               'Philadelphia Story, The (1940)']},
                 {'Genres': 'Sci-Fi', 'Top5_most_rating_id': [260, 1196, 1210, 480, 589], 'Top5_highest_rating_id': [260, 750, 2571, 1196, 541],
                  'Top5_most_rating_name': ['Star Wars: Episode IV - A New Hope (1977)', 'Star Wars: Episode V - The Empire Strikes Back (1980)',
                                            'Star Wars: Episode VI - Return of the Jedi (1983)', 'Jurassic Park (1993)',
                                            'Terminator 2: Judgment Day (1991)'],
                  'Top5_highest_rating_name': ['Star Wars: Episode IV - A New Hope (1977)',
                                               'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)', 'Matrix, The (1999)',
                                               'Star Wars: Episode V - The Empire Strikes Back (1980)', 'Blade Runner (1982)']},
                 {'Genres': 'Thriller', 'Top5_most_rating_id': [589, 2571, 593, 608, 2762], 'Top5_highest_rating_id': [745, 50, 904, 1212, 2762],
                  'Top5_most_rating_name': ['Terminator 2: Judgment Day (1991)', 'Matrix, The (1999)', 'Silence of the Lambs, The (1991)',
                                            'Fargo (1996)', 'Sixth Sense, The (1999)'],
                  'Top5_highest_rating_name': ['Close Shave, A (1995)', 'Usual Suspects, The (1995)', 'Rear Window (1954)', 'Third Man, The (1949)',
                                               'Sixth Sense, The (1999)']},
                 {'Genres': 'War', 'Top5_most_rating_id': [1196, 1210, 2028, 110, 527], 'Top5_highest_rating_id': [527, 1178, 750, 912, 1204],
                  'Top5_most_rating_name': ['Star Wars: Episode V - The Empire Strikes Back (1980)',
                                            'Star Wars: Episode VI - Return of the Jedi (1983)', 'Saving Private Ryan (1998)', 'Braveheart (1995)',
                                            "Schindler's List (1993)"],
                  'Top5_highest_rating_name': ["Schindler's List (1993)", 'Paths of Glory (1957)',
                                               'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)', 'Casablanca (1942)',
                                               'Lawrence of Arabia (1962)']},
                 {'Genres': 'Western', 'Top5_most_rating_id': [590, 1304, 2012, 3671, 1266], 'Top5_highest_rating_id': [3607, 3030, 1304, 1283, 1201],
                  'Top5_most_rating_name': ['Dances with Wolves (1990)', 'Butch Cassidy and the Sundance Kid (1969)',
                                            'Back to the Future Part III (1990)', 'Blazing Saddles (1974)', 'Unforgiven (1992)'],
                  'Top5_highest_rating_name': ['One Little Indian (1973)', 'Yojimbo (1961)', 'Butch Cassidy and the Sundance Kid (1969)',
                                               'High Noon (1952)', 'Good, The Bad and The Ugly, The (1966)']}]



def api_rating(request):
	"""

	from pprint import pprint
	from PyHelpers import set_logging; set_logging(10)

	import os
	os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MovieAppBackendServer.settings")

	from django.test import RequestFactory
	from django.conf import settings

	settings.configure()
	request_factory = RequestFactory()

	genre_name = 'Action'
	page = 1
	request = request_factory.get(f'http://127.0.0.1:7000/api_genre?rating1={rating1}&movie1={movie1}')

	:param request:
	:return:
	"""
	logging.info(request.GET)

	rating1 = request.GET.get('rating1')
	movie1 = request.GET.get('movie1')
