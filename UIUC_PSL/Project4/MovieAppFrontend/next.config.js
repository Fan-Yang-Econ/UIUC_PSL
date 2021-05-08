/* eslint-disable */
const withCss = require('@zeit/next-css')
const withLess = require('@zeit/next-less')

// Support Antd
// https://github.com/zeit/next.js/blob/canary/examples/with-ant-design/next.config.js
module.exports = withCss(withLess({
	// cssModules: true,
	// cssLoaderOptions: {
	//   sourceMap: false,
	//   importLoaders: 1,
	// },
	lessLoaderOptions: {
		javascriptEnabled: true
	},
	typescript: {
		// !! WARN !!
		// Dangerously allow production builds to successfully complete even if
		// your project has type errors.
		// !! WARN !!
		ignoreBuildErrors: true,
	},
	webpack: (config, {isServer}) => {
		if (isServer) {
			const antStyles = /antd\/.*?\/style\/css.*?/
			const origExternals = [...config.externals]
			config.externals = [
				(context, request, callback) => {
					if (request.match(antStyles)) return callback()
					if (typeof origExternals[0] === 'function') {
						origExternals[0](context, request, callback)
					} else {
						callback()
					}
				},
				...(typeof origExternals[0] === 'function' ? [] : origExternals),
			]

			config.module.rules.unshift({
				test: antStyles,
				use: 'null-loader',
			})
		}
		return config
	},
}))


// module.exports = withCss({
//   webpack: (config, { isServer }) => {
//     if (isServer) {
//       const antStyles = /antd\/.*?\/style\/css.*?/
//       const origExternals = [...config.externals]
//       config.externals = [
//         (context, request, callback) => {
//           if (request.match(antStyles)) return callback()
//           if (typeof origExternals[0] === 'function') {
//             origExternals[0](context, request, callback)
//           } else {
//             callback()
//           }
//         },
//         ...(typeof origExternals[0] === 'function' ? [] : origExternals),
//       ]
//
//       config.module.rules.unshift({
//         test: antStyles,
//         use: 'null-loader',
//       })
//     }
//     return config
//   },
// })
