module.exports = {
	'env': {
		'browser': true,
		'es6': true
	},
	'extends': [
		'eslint:recommended',
		'plugin:react/recommended'
	],
	'globals': {
		'Atomics': 'readonly',
		'SharedArrayBuffer': 'readonly',
		"require": true
	},
	"ignorePatterns": ["**/*.ts"],
	"parser": "babel-eslint",
	'parserOptions': {
		'ecmaFeatures': {
			'jsx': true,
			"experimentalObjectRestSpread": true,
		},

		'ecmaVersion': 2018,
		'sourceType': 'module'
	},
	'plugins': [
		'react'
	],
	'rules': {
		'indent': [
			'off',
			'tab'
		],
		"react/prop-types":[
			'off'
		],
		"react/display-name":[
			'off'
			],
		"no-mixed-spaces-and-tabs": [
			"warn",
			"smart-tabs"
		],
		'linebreak-style': [
			'error',
			'unix'
		],
		'semi': [
			'error',
			'never'
		]
	}
};
