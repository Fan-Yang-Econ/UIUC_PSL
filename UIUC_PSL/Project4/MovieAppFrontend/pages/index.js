import React, {useState} from 'react'
import Head from 'next/head'

import {Button, Col, Row, message, Space, Tag, Layout, Rate} from 'antd'

const {Header, Content} = Layout;
import {Menu} from 'antd';
import {HeartOutlined, StarOutlined, SettingOutlined} from '@ant-design/icons';

const {SubMenu} = Menu;


const DEFAULT_RATING = 5
const NUMBER_OF_MOVIES_TO_RATE = 5

function getRandomMovieId(max = 3952) {
	return Math.floor(Math.random() * max);
}

const LIST_GENRES = [
	"Action",
	"Adventure",
	"Animation",
	"Children's",
	"Comedy",
	"Crime",
	"Documentary",
	"Drama",
	"Fantasy",
	"Film-Noir",
	"Horror",
	"Musical",
	"Mystery",
	"Romance",
	"Sci-Fi",
	"Thriller",
	"War",
	"Western"
]


const PAGE_WELCOME = 'WELCOME'
const PAGE_RATING = 'RATING'
const PAGE_GENRE = 'GENRE'


const get_genre = async (genre_name) => {

	const result = await fetch('/api/call_backend',
		{
			'method': 'POST',
			'body': JSON.stringify({'genre_name': genre_name}),
		})
		.then(response => {
				console.log(response)
				return response.json()
			}
		)
		.catch(error => {
			console.log("API Error")
			console.log(error)
		})

	return result
}

const get_image_title = (movie_id, df_movie_json) => {

	if (Number(movie_id) in df_movie_json) {
		return df_movie_json[Number(movie_id)]['Title']
	} else {
		return 'Unknown'
	}
}

const _get_movies_to_rate = (df_movie_json, max_number_movies_to_rate = NUMBER_OF_MOVIES_TO_RATE) => {

	const _state_MoviesToRate = []
	while (_state_MoviesToRate.length < max_number_movies_to_rate) {
		const random_id = getRandomMovieId()
		console.log('tried', random_id)
		if (random_id in df_movie_json && !_state_MoviesToRate.includes(random_id)) {
			_state_MoviesToRate.push(random_id)
		}
	}

	return _state_MoviesToRate
}

// style={{minHeight:'100%', position:'absolute', minWidth:'100%'}}
const Landing = props => {

	const [state_IMAGE_for_Genre, set_Image_for_genre] = useState([]);
	const [state_RATINGS, set_rating] = useState({});
	const [state_MoviesToRate, set_MoviesToRate] = useState(_get_movies_to_rate(props.df_movie_json))
	const [state_Page, set_Page] = useState(PAGE_WELCOME)
	const [state_MoviesRecommended, set_MoviesRecommended] = useState([])


	const refresh_movies_to_rate = () => {
		const movies_to_rate = _get_movies_to_rate(props.df_movie_json)
		console.log('Refresh movies to rate', )
		set_MoviesToRate(movies_to_rate)
		set_MoviesRecommended([])
	}

	const store_rating = (id, rating) => {
		console.log(`Move ${id} is rated as ${rating}`)
		state_RATINGS[id] = rating
		set_rating({...state_RATINGS})
	}

	const submit_rating = async () => {

		const ratings_to_submit = {}
		state_MoviesToRate.map(movie_id => {
			if (movie_id in state_RATINGS) {
				ratings_to_submit[movie_id] = state_RATINGS[movie_id]
			} else {
				ratings_to_submit[movie_id] = DEFAULT_RATING
			}
		})

		console.log('submit_rating', ratings_to_submit)


		const result = await fetch('/api/call_backend',
			{
				'method': 'POST',
				'body': JSON.stringify({'ratings_to_submit': ratings_to_submit}),
			})
			.then(response => {
					console.log(response)
					return response.json()
				}
			)
			.catch(error => {
				console.log("API Error")
				console.log(error)
			})

		console.log('recommended result', result)

        if (result === undefined){
            message.warn('No movies can be found')
            set_MoviesRecommended([])
        } else {
            set_MoviesRecommended(result)
        }


	}
	const select_a_menu = async (event) => {

		console.log(event)

		if (event['key'] === PAGE_RATING) {
			set_Page(PAGE_RATING)
		} else if (LIST_GENRES.includes(event['key'])) {
			message.info(`We recommend the following 5 movies with the genre ${event['key']}`);
			let list_suggested_by_genre = await get_genre(event['key'])

            if (list_suggested_by_genre === undefined){
                message.warn('No movies can be found')
            } else {
                set_Image_for_genre(list_suggested_by_genre)
                set_Page(PAGE_GENRE)
            }
		}

	}

	return (
		<div>
			<Head>
				<link rel="shortcut icon" type="image/png" href={"/favicon-32x32.png"}/>
				<title>Movie Recommendation App (Authors: fanyang3 & xiaozhu3)</title>
				<meta charSet="utf-8"/>
				<meta name="viewport" content="initial-scale=1.0, width=device-width"/>
			</Head>
			<Layout>
				<Header><p style={{color: '#7ec1ff'}}><img src={`/favicon-32x32.png`}></img> The Movie Recommendation App</p></Header>
				<Content style={{backgroundColor: 'white'}}>
					<Menu mode="horizontal" onSelect={select_a_menu}>
						<SubMenu key="Select Genre" icon={<HeartOutlined/>} title={"Recommend by Genre"}>
							{LIST_GENRES.map(genre => {
								return <Menu.Item key={genre}>{genre}</Menu.Item>
							})
							}
						</SubMenu>
						<Menu.Item key={PAGE_RATING} icon={<StarOutlined/>}>
							Recommend by Similar Ratings
						</Menu.Item>
					</Menu>
					{state_Page === PAGE_WELCOME &&
					<Row justify="center" style={{marginTop: '270px', fontSize: '30px'}}>
						<Col span={24} style={{textAlign: 'center'}}>
							<p> Welcome to our Movie Recommendation App</p>
							<img src={`/MovieImages/1721.jpg`}></img>
						</Col>
					</Row>
					}

					{state_Page === PAGE_GENRE &&
					<Row style={{marginLeft: '100px', marginTop: '50px'}} gutter={[30, 40]}>
						{state_IMAGE_for_Genre.length > 0 && state_IMAGE_for_Genre.map(image_id => {
							return <Col span={6}>
								<img key={`genre_${image_id}`} src={`/MovieImages/${image_id}.jpg`} alt={`movie id ${image_id}`}/>
								<p>{get_image_title(image_id, props.df_movie_json)}</p>
							</Col>
						})
						}

					</Row>
					}

					{state_Page === PAGE_RATING && state_MoviesRecommended.length === 0 &&
					<Row style={{marginLeft: '30px'}}>
						<Col span={24} style={{marginTop: '10px', marginBottom: '50px'}}>
							<p style={{fontSize: '20px'}}>
								Please rate the following movies. Then we will recommend you some movies based on your ratings.</p>
							<Space>
								<Button type="primary" onClick={refresh_movies_to_rate}>Refresh</Button>
								<Button type="primary" onClick={submit_rating}>Submit</Button>
							</Space>

						</Col>

						{state_MoviesToRate.map(image_id =>
							<Col span={6} key={`rating_for_${image_id}`}>
								<img src={`/MovieImages/${image_id}.jpg`} alt={`${image_id}`}/>
								<p>{get_image_title(image_id, props.df_movie_json)}</p>
								<Rate defaultValue={DEFAULT_RATING} onChange={rating => store_rating(image_id, rating)}/>
							</Col>
						)}
					</Row>
					}

					{state_Page === PAGE_RATING && state_MoviesRecommended.length > 0 &&
					<Row style={{marginLeft: '30px', marginBottom: '100px'}} gutter={[0, 10]}>
						<Col span={24} style={{marginTop: '10px', marginBottom: '50px'}}>
							<p style={{fontSize: '20px'}}>
								Please see the following recommended movies based on your ratings.</p>
							<Space>
								<Button type="primary" onClick={refresh_movies_to_rate}>Refresh</Button>
							</Space>
						</Col>
						<Row gutter={[0, 100]}>
							{state_MoviesRecommended.map(result => {
									const movie_id = result['MovieID']
									return <Col span={6} key={`recommend_for_${movie_id}`}>
										<img src={`/MovieImages/${movie_id}.jpg`} alt={`${movie_id}`}/>
										<p>{get_image_title(movie_id, props.df_movie_json)}</p>

										{result['rating_source'] === 'new' &&

										<Tag color="red-inverse">You liked a similar movie</Tag>

										}

										{result['rating_source'] === 'old' &&
										<Tag color="purple-inverse">Other users like this movie</Tag>
										}

									</Col>
								}
							)}
						</Row>
					</Row>
					}

				</Content>
			</Layout>
		</div>
	)
}

export async function getStaticProps(context) {

	const df_movie_json = await import('public/df_movie_names.json')

	const props = {
		'df_movie_json': JSON.parse(JSON.stringify(df_movie_json))
	}
	return {
		props, // will be passed to the page component as props
	}
}


export default Landing


