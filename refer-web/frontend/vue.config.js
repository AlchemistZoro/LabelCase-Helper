const Components = require('unplugin-vue-components/webpack')
const {VarletUIResolver, ArcoResolver} = require('unplugin-vue-components/resolvers')

module.exports = {
  configureWebpack: {
    plugins: [
      Components({
        resolvers: [VarletUIResolver(), ArcoResolver()]
      })
    ]
  },
}
