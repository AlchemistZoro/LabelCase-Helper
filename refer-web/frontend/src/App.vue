<template>
  <div class="background"/>
  <transition name="bloom" appear>
    <var-card class="container card" elevation="6">
      <template #description>
        <div class="wrap">
          <div class="left">
            <div class="title">案情分类小助手</div>
            <div style="height: 64vh;display: flex;justify-content: space-between;flex-direction: column">
              <a-upload style="height: 15vh;" :custom-request="customRequest" accept="text/plain"
                        draggable/>
              <Editor
                autofocus
                style="height: 46vh;"
                class="input"
                :class="{focused:focused}"
                v-model="valueHtml"
                :defaultConfig="editorConfig"
                :mode="mode"
                @onCreated="handleCreated"
                @focus="focused=true"
                @blur="focused=false"
              />
            </div>
            <div class="button">
              <var-button :disabled="editor?.isEmpty()||loading" type="success" @click="send">确定</var-button>
              <var-button :disabled="editor?.isEmpty()||loading" type="danger" @click="clear">清空</var-button>
            </div>
          </div>

          <div class="right var-elevation--5">
            <div class="category">所属分类</div>
            <transition name="slide" appear>
              <div class="tree" v-if="Object.keys(temp).length">
                <transition name="bloom" appear>
                  <a-tree
                    :blockNode="true"
                    :checkable="false"
                    :data="data"
                    @select="highlight"
                  >
                    <template #extra="nodeData">
                      <var-icon style="border: 1px solid #3370ff;border-radius: 3px;margin-right: 3px" size="16"
                                color="#3370ff"
                                v-if="!nodeData.end" name="plus" @click="add(nodeData)"/>
                      <var-icon style="border: 1px solid red;border-radius: 3px;margin-right: 3px" size="16" color="red"
                                name="minus"
                                @click="remove(nodeData)"/>
                    </template>
                  </a-tree>
                </transition>
              </div>
            </transition>
            <div class="tip" v-if="!uploaded">
              分类结果将显示在这
            </div>
            <div class="tip" v-if="uploaded&&loading">
              分析中...
            </div>
            <div style="display:flex;justify-content: space-between" v-if="Object.keys(temp).length">
              <var-button class="download" type="info" @click="download_excel">导出excel</var-button>
              <var-button class="download" type="info" @click="download_txt">导出txt</var-button>
            </div>
          </div>
        </div>
      </template>
    </var-card>
  </transition>

  <var-popup style="border-radius: 10px" v-model:show="show" @close="new_category={
          first: null,
          second: null,
          third: null,
        }">
    <div class="add-text">添加类别</div>
    <div class="new-category">
      <var-select class="option" clearable placeholder="请选择一个选项" v-model="new_category.first">
        <var-option v-for="i in Object.keys(category)" :label="i"/>
      </var-select>
      <var-select class="option" clearable :disabled="!new_category.first" placeholder="请选择一个选项"
                  v-model="new_category.second">
        <var-option v-if="new_category.first" v-for="i in Object.keys(category[new_category.first])" :label="i"/>
      </var-select>
      <var-select class="option" clearable :disabled="!new_category.second || !new_category.first" placeholder="请选择一个选项"
                  v-model="new_category.third">
        <var-option v-if="new_category.first && new_category.second"
                    v-for="i in category[new_category.first][new_category.second]" :label="i"/>

      </var-select>
    </div>
    <div class="add-button">
      <var-button
        :disabled="!new_category.first||!new_category.second||!new_category.third"
        type="success"
        @click="add_category">确定
      </var-button>
    </div>
  </var-popup>
</template>

<script>
  import {category} from "./assets/js/category";
  import '@wangeditor/editor/dist/css/style.css' // 引入 css
  import {Editor} from '@wangeditor/editor-for-vue'

  export default {
    name: "App",
    components: {
      Editor
    },
    data() {
      return {
        files: [],
        valueHtml: "",
        editorConfig: {
          hoverbarKeys: {
            text: []
          },
          autoFocus: false,
          placeholder: '请输入内容...',
        },
        editor: null,
        mode: 'default',
        api: "ws://api.seutools.com:7002/",
        focused: false,
        temp: {},
        uploaded: false,
        loading: false,
        show: false,
        category: category,
        new_category: {
          first: null,
          second: null,
          third: null,
        },
        lines: {},
        p: [],
        oldHtml: null,
        highlights: []
      }
    },
    computed: {
      data() {
        return Object.keys(this.temp).map(x => {
          return {
            title: x,
            key: x,
            path: [x],
            children: Object.keys(this.temp[x]).map(y => {
              return {
                title: y,
                key: x + y,
                path: [x, y],
                children: this.temp[x][y].map(z => {
                  return {
                    title: z,
                    key: x + y + z,
                    path: [x, y, z],
                    end: true
                  }
                })
              }
            }),
          }
        })
      }
    },
    methods: {
      download_excel() {
        let res = []
        for (let i of Object.keys(this.temp)) {
          for (let j of Object.keys(this.temp[i])) {
            for (let k of this.temp[i][j]) {
              res.push([i, j, k])
            }
          }
        }
        const excel = require("xlsx")
        const sheet1 = excel.utils.aoa_to_sheet(res);

        const wb = excel.utils.book_new();
        excel.utils.book_append_sheet(wb, sheet1, '分类详情');
        excel.writeFile(wb, '分类.xlsx')
      },
      download_txt() {
        let aTag = document.createElement('a');
        aTag.download = "分类.txt";

        let res = []
        for (let i of Object.keys(this.temp)) {
          for (let j of Object.keys(this.temp[i])) {
            for (let k of this.temp[i][j]) {
              res.push(i + "/" + j + "/" + k + "\n")
            }
          }
        }
        let blob = new Blob(res, {
          type: "text/plain;charset=utf-8"
        });
        aTag.href = URL.createObjectURL(blob);
        aTag.click();
      },
      customRequest(option) {
        const {fileItem} = option
        let reader = new FileReader();
        let that = this
        reader.onload = function (ev) {
          that.clear()
          that.editor.insertText(reader.result)
        };
        reader.readAsText(fileItem.file);
      },

      highlight(data) {
        let html = this.oldHtml
        for (let i of this.lines[data]) {
          html = html.replace(this.p[i], `<span style=\"background-color: rgb(255, 255, 0);\" id='anchor-${i}'> ${this.p[i]} </span>`)
        }
        this.editor.setHtml(html)
      },
      handleCreated(editor) {
        this.editor = Object.seal(editor)
      },
      remove(data) {
        switch (data.path.length) {
          case 1:
            delete this.temp[data.path[0]]
            break
          case 2:
            delete this.temp[data.path[0]][data.path[1]]
            break
          case 3:
            this.temp[data.path[0]][data.path[1]].splice(this.temp[data.path[0]][data.path[1]].indexOf(data.path[2]), 1)
            break
        }
        this.$tip({
          content: "删除成功",
          type: "success",
          duration: 1000
        })
      },
      add(data) {
        this.new_category.first = data.path[0]
        this.new_category.second = data.path[1]
        this.show = true
      },
      add_category() {
        let labels = Object.values(this.new_category)
        if (this.temp[labels[0]]) {
          if (this.temp[labels[0]][labels[1]]) {
            if (this.temp[labels[0]][labels[1]].indexOf(labels[2]) === -1) {
              this.temp[labels[0]][labels[1]].push(labels[2])
            }else {
              this.$tip({
                content: "该分类已有",
                type: "warning",
                duration: 1000
              })
              return
            }
          } else {
            this.temp[labels[0]][labels[1]] = [labels[2]]
          }
        } else {
          this.temp[labels[0]] = {[labels[1]]: [labels[2]]}
        }
        this.$tip({
          content: "添加成功",
          type: "success",
          duration: 1000
        })
        this.show = false
      },
      clear() {
        this.editor.clear()
        this.temp = {}
        this.uploaded = false
      },
      merge_arr(arr1, arr2) {
        let _arr = [];
        for (let i = 0; i < arr1.length; i++) {
          _arr.push(arr1[i]);
        }
        for (let i = 0; i < arr2.length; i++) {
          let flag = true;
          for (let j = 0; j < arr1.length; j++) {
            if (arr2[i] === arr1[j]) {
              flag = false;
              break;
            }
          }
          if (flag) {
            _arr.push(arr2[i]);
          }
        }
        return _arr;
      },
      send() {
        this.uploaded = true
        this.temp = {}
        const ws = new WebSocket(this.api)
        const that = this
        ws.onopen = function () {
          console.log('Connection open ...');
          that.loading = true
          that.$tip({
            content: "分析中",
            type: "success",
            duration: 1000
          })
          that.p = that.editor.getText().split(/[。；;\n]/)
          ws.send(JSON.stringify({lines: that.p}))
        };

        ws.onmessage = function (e) {
          that.lines = {}
          that.temp = {}
          that.oldHtml = that.valueHtml
          let res = JSON.parse(e.data)
          let lines, labels
          for (let label of Object.keys(res)) {
            console.log(label);
            lines = res[label]
            labels = label.split("/")

            if (that.lines[labels[0]]) {
              that.lines[labels[0]] = that.merge_arr(that.lines[labels[0]], lines)
            } else {
              that.lines[labels[0]] = lines
            }

            if (that.lines[labels[0] + labels[1]]) {
              that.lines[labels[0] + labels[1]] = that.merge_arr(that.lines[labels[0] + labels[1]], lines)
            } else {
              that.lines[labels[0] + labels[1]] = lines
            }

            if (that.lines[labels[0] + labels[1] + labels[2]]) {
              that.lines[labels[0] + labels[1] + labels[2]] = that.merge_arr(that.lines[labels[0] + labels[1] + labels[2]], lines)
            } else {
              that.lines[labels[0] + labels[1] + labels[2]] = lines
            }

            if (that.temp[labels[0]]) {
              if (that.temp[labels[0]][labels[1]]) {
                that.temp[labels[0]][labels[1]].push(labels[2])
              } else {
                that.temp[labels[0]][labels[1]] = [labels[2]]
              }
            } else {
              that.temp[labels[0]] = {[labels[1]]: [labels[2]]}
            }
          }
          console.log(that.lines);
        };

        ws.onclose = function () {
          that.$tip({
            content: "分析完成",
            type: "success",
            duration: 1000
          })
          that.loading = false
          console.log('Connection closed.');
        };

      }
    },
    beforeUnmount() {
      const editor = this.editor
      if (editor == null) return
      editor.destroy()
    }
  }
</script>

<style>
  @import "assets/css/normalize.css";
  @import "assets/css/transition.css";

  @media screen and (min-width: 840px) {
    .container {
      width: 60vw;
      height: 80vh;
      background-color: #aaaaaa;
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      min-width: 800px
    }

    .wrap {
      display: flex;
      justify-content: space-between;
    }

    .left {
      width: 65%;
      padding: 20px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }

    .title {
      font-size: 20px;
      font-weight: bold;
      margin-bottom: 10px;
    }

    .input {
      height: 64vh;
      border-radius: 5px;
      border: 2px solid #f1f1f1;
    }

    .button {
      padding: 20px 0;
      display: flex;
      justify-content: space-between;
    }

    .right {
      width: 35%;
      height: 80vh;
      background-color: #f5f5dc;
    }

    .tree {
      border-top: 1px solid #aaaaaa;
      border-bottom: 1px solid #aaaaaa;
      height: 64vh;
      overflow-y: scroll;
      margin: 0 10px;
      background-color: rgba(255, 255, 255, .5);
    }

    .category {
      margin-top: 20px;
      font-size: 20px;
      font-weight: bold;
      text-align: center;
      margin-bottom: 10px;
    }

    .tip {
      margin: 30px 0;
      text-align: center;
      color: #aaaaaa;
    }

    .loading {
      text-align: center;
      color: #aaaaaa;
    }

    .add-text {
      line-height: 40px;
      margin-top: 20px;
      font-size: 20px;
      font-weight: bold;
      text-align: center;
    }

    .new-category {
      width: 50vw;
      padding: 20px;
      display: flex;
      justify-content: space-around;
    }

    .option {
      width: 15vw;
    }

    .add-button {
      padding: 20px;
      float: right;
    }

    .download {
      margin: 20px 10px;
      float: right;
    }
  }


  .card {
    --card-footer-margin: 0;
    --card-padding: 0;
    --card-footer-padding: 0;
    border-radius: 20px;
  }


  .background {
    background: url("assets/img/background.jpg");
    opacity: 0.9;
    background-size: 100% auto;
    position: fixed;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
  }


  .focused {
    border: 2px solid #4ebaee;
  }

</style>
