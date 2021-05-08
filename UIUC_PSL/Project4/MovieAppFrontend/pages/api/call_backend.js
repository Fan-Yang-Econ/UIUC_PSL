// see https://nextjs.org/docs/api-routes/introduction

export default async (req, res) => {
	let body_dict = JSON.parse(req['body'])

	let fullUrl
	if ('genre_name' in body_dict) {
		const genre_name = body_dict['genre_name']
		const api_genre = process.env.NEXT_PUBLIC_BACKEND_API + 'api_genre'
		fullUrl = `${api_genre}?genre_name=${genre_name}`
	} else if ('ratings_to_submit' in body_dict) {
		const ratings_to_submit = body_dict['ratings_to_submit']
		fullUrl = process.env.NEXT_PUBLIC_BACKEND_API + 'api_rating?'
		Object.keys(ratings_to_submit).map((movie_id, count_i) => {
			fullUrl += `movie${count_i}=${movie_id}&rating${count_i}=${ratings_to_submit[movie_id]}&`
		})

		fullUrl = fullUrl.replace(/&$/, '')

	}

	console.log('fullUrl in API', fullUrl)
	const response = await fetch(fullUrl, {'mode': 'cors'})
	if (response.ok) { // if HTTP-status is 200-299
		// get the response body (the method explained below)
		console.log('response', response)
		const r_json = await response.json()
		res.statusCode = 200
		res.setHeader('Content-Type', 'application/json')
		res.send(r_json)

	} else {
		console.log("HTTP-Error")
		console.log(response) // false
		res.statusCode = 500
		res.setHeader('Content-Type', 'application/json')
		res.send({'error': true})
	}

}


