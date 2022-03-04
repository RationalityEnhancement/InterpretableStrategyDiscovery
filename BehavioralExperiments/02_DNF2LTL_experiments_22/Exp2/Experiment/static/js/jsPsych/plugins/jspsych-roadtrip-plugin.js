
jsPsych.plugins['roadtrip-plugin'] = (function(){


  var plugin = {};

  plugin.info = {
    name: 'roadtrip-plugin',
    parameters: {
    }
  }

  plugin.trial = function(display_element, trialConfig){

    trial = new Roadtrip(display_element, trialConfig);

    //setTimeout(function(){jsPsych.finishTrial();},  5000);
  }




  Roadtrip = class Roadtrip{
    constructor(display_element, trial_info) {

      //Information needed to run trial
      this.display = display_element;
      this.hide_feedback_timer= 0;
      this.time_cost_interval= 0;
      this.num_practice_trials =0

      this.data_log = []
      this.map = {
        nodes: {}, // vis.DataSet()
        edges: {}, // vis.DataSet()
        net: {}, // vis.Network()
      };
      this.is_selected = {}
      this.node_border_width = 2;
      this.selected_edge_color = "#A40802"
      this.selected_node_color = "#A40802"
      this.unselected_edge_color = "#848484"
      this.unselected_edge_width = 1;
      this.selected_edge_width = 2;
      this.node_on_route_width = 3;
      this.edge_on_route_width = 3;
      this.use_end_city_reward = true;
      this.time_cost = 0;
      this.time_penalty = 10;
      this.reveal_delay = 1000;
      this.lookup_cost = 0;
      this.lookup_fee = 10;
      this.confirmed = false;

      // vis.js network options
      this.vis_options = {
        physics: {
          enabled: false
        },
        edges: {
          smooth: {
            enabled: false
          },
          color: {
            color: this.unselected_edge_color,
            hover: this.selected_edge_color,
            highlight: this.selected_edge_color,
            opacity: 1,
          },
          width: this.unselected_edge_width,
          hoverWidth: this.unselected_edge_width,
          arrows:'middle'
        },
        nodes : {
          labelHighlightBold: false,
          shape: "dot",
          size: 10,
          borderWidth: this.node_border_width,
          font: {
            strokeWidth : 5
          },
          color: {
            border: "#2471A3",
            background: "#E5E7E9",
            highlight:{
              border: "#2471A3",
              background: "#CACFD2",
            },
            hover:{
              border: "#2471A3",
              background: "#CACFD2",
            }
          }
        },
        interaction: {
          dragView: false,
          dragNodes: false,
          selectConnectedEdges: false,
          hover: true,
          hoverConnectedEdges: false,
          zoomView: false,
          selectable: true,
          keyboard: {
            enabled: false
          },
          multiselect: false
        },
        manipulation: {
          enabled: false,
          addNode: false,
          deleteNode: false,
          addEdge: function(edgeData,callback) {
            if (edgeData.from !== edgeData.to) {
              callback(edgeData);
            }
          }
        }
      };

      this.city_info  = trial_info['city_info'];
      this.graph_data = trial_info['graph'];
      this.start_city = trial_info['start_city'];
      this.end_cities = trial_info['end_cities'];
      this.map_name =   'static/images/roadtrip/maps/' + trial_info['map'];
      this.city_names = Object.keys(this.city_info);
      this.is_practice = false;


      //LOG
      for(let city_name in this.city_info){
        let pr = this.city_info[city_name]["possible_rewards"];
        this.city_info[city_name]['reward'] = pr[Math.floor(Math.random() * pr.length)];
      }

      let nodes = this.graph_data.nodes;
      for(let node in nodes){
        nodes[node]['reward'] = this.city_info[nodes[node]['label']]["reward"];
        nodes[node]['inEdge'] = [];
        nodes[node]['outEdge'] = [];
      }
      for(let edge of this.graph_data.edges){
        nodes[edge[0]]['outEdge'].push(edge[1]);
        nodes[edge[1]]['inEdge'].push(edge[0]);
      }

      this.data = {
        'nodes': nodes,
        'start_nodes': this.name2id(this.start_city),
        'end_nodes': this.name2id(this.end_cities),
        'clicks': [],
        'cost_total': 0,
        'cost_path': 0,
        'cost_expected': 0,
        'path': []
      }

      //INIT
      // create dom
      this.htmlToElements = function(container, html) {
          var template = document.createElement('template');
          template.innerHTML = html;
          container.append(template.content.childNodes);
      }

      this.htmlToElements(this.display, "<div id='map'> <div id='canvas'></div> </div> <div id='cityinput' autocomplete='off'> <input type='text' placeholder='City'></input> <input type='submit' value='Reveal'></input> </div> <div id='feedbackroadtrip'><p style='font-size:3vh;text-align: center;'></p></div> <div id='timecost'><p style='font-size:3vh;text-align: center;'></p></div>  <div id='submitnext'><button id='finishedbutton' class='mybutton'>Submit route</button> <button id='nexttrialbutton' class='mybutton'>Next</button></div> <p id = 'car' class='myicon' style = 'display:none'><i class='fas fa-car-side'></i></p>");

      var image = new Image();
      image.src = this.map_name;
      this.enable_resize_correction(image.width/image.height);
      this.set_map_image(this.map_name);
      this.draw_graph_from_json(this.graph_data);
      this.draw_car();
      this.set_autocomplete(true);
      this.enable_report_mode();
      this.set_lookup_cost(0);


      var self = this;
      $("#finishedbutton").off("click").on("click",function(){

        var route = self.get_unambiguous_route(self.start_city, self.end_cities);
        console.log(route);

        if(!self.end_cities.includes(route[route.length-1])){
          self.show_feedback("Error: Invalid route",2000);

        } else {

          //self.stop_time_cost_interval();
          $("#finishedbutton").off("click");

          self.show_route(route, true, self.is_practice, function(){

            //end Trial --> rewrite
            $("#nexttrialbutton").show().off('click').on('click',function(){
              //self.reset();

              $("#feedbackroadtrip").hide();
              $("#nexttrialbutton").hide();
              self.map.net.destroy();
              $("#cityinput [type=text]").val('');
              $('.plane').remove();
              $('#timecost').hide();
              self.map = {
                nodes: {}, // vis.DataSet()
                edges: {}, // vis.DataSet()
                net: {}, // vis.Network()
              };

              //end of block
              //jsPsych.finishTrial(this.data);
              //self.display.empty();
              //return jsPsych.endCurrentTimeline();

              //end of trials
              jsPsych.finishTrial(self.data);
              return self.display.empty();
            });
          })

      }
      });


  };

    shuffle(array){
      var currentIndex = array.length, temporaryValue, randomIndex;
      while (0 !== currentIndex){
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
      }
      return array;
    }

    get_graph_data(){
      this.map.net.storePositions();
      var x = JSON.parse(JSON.stringify(this.map.nodes._data))
      _.forEach(x,function(val,key,obj){
        obj[key].x = obj[key].x/$("#canvas").width()
        obj[key].y = obj[key].y/$("#canvas").height()
      })
      return {
        edges : _.this.map(this.map.edges._data, function(d,idx){return [d.from, d.to];}),
        nodes : x
      }
    }

    set_map_image(name){
      $("#map").css('background-image',"url('" + name + "')")
    }

    get_map_name(){
      var name = $("#map").css('background-image').replace(/url\(['"]*(.*?)['"]*\)/g, '$1')
      return name.slice(name.indexOf('fantasy'))
    }

    get_trial_info(){
      return {"this.city_info" : this.city_info, "map" : this.get_map_name(), "graph" : this.get_this.graph_data(), "start_city" : this.start_city, "this.this.end_cities" : this.end_cities}
    }

    update_edge_colors(){
      let self = this;
      _.forEach(this.is_selected,function(val,key,obj){
        self.map.edges.update({
          id : key,
          color: val ? self.selected_edge_color : self.unselected_edge_color,
          width: val ? self.selected_edge_width : self.unselected_edge_width
        })
      })
    }

    show_feedback(feedbacktext,duration,callback, color='black'){
      //log_data({"event_type": "Show feedback", "event_info" : {"feedback" : feedbacktext}})
      $('#feedbackroadtrip').show()
      $('#feedbackroadtrip p').text(feedbacktext);
      $('#feedbackroadtrip p').css('color', color);
      clearTimeout(this.hide_feedback_timer);
      if(duration != undefined){
        this.hide_feedback_timer = setTimeout(function(){
          $('#feedbackroadtrip').hide()
          if(callback != undefined){
            callback()
          }
        },duration)
      }
    }

    show_reward_on_city(id, city_name, feedback, log=false){
      var reward = this.get_reward(city_name);
      if(reward==undefined) reward = this.city_info[city_name]["possible_rewards"][Math.floor(Math.random() * 4)];

      //log
      if(log) this.data.clicks.push(id);
      this.map.nodes.update({id : id, label: "\n" + city_name + "\n-$" + reward})

      //feedback
      if(feedback){
        this.show_feedback(city_name + ": -$" + reward,1000)
      }
    }

    is_reward_revelaed(id){
      let label = this.map.nodes.get(id).label;
      label = label.split('\n');
      return (label.length > 1);
    }

    clean_label(label){
      return label.toString().split('\n').join('').split('-$')[0]
    }

    get_reward(label){
      /*let l = label.split('\n');
      if(l.length>1) return parseFloat(l[l.length-1].replace('$',''));
      return undefined */

      // handle revealed nodes
      let l = label.split('\n');
      console.log(l);
      if(l.length > 1) label = l[1];
      return this.city_info[label]["reward"];
    }

    get_average_reward(city_name){
      var retval =0
      for(var i=0;i<this.city_info[city_name]["possible_rewards"].length;i++){
        retval += this.city_info[city_name]["possible_rewards"][i]/this.city_info[city_name]["possible_rewards"].length
      }
      return retval
    }

    name2id(city_names){
      let res = [];
      if(!Array.isArray(city_names)) city_names = [city_names];

      for(let c of city_names){
        for(let n in this.graph_data.nodes){
          if(this.graph_data.nodes[n].label == c){
            res.push(this.graph_data.nodes[n].id);
            break;
          }
        }
      }
      return res
    }

    start_time_cost_interval(){
      this.time_cost = 0
      $('#timecost').show();
      $('#timecost p').text('Time cost: $' + this.time_cost.toString());

      let self = this;
      this.time_cost_interval = setInterval(function(){
        self.time_cost += self.time_penalty;
        $('#timecost p').text('Time cost: $' + self.time_cost.toString());
      },1000)
    }

    stop_time_cost_interval(){
      clearInterval(this.time_cost_interval)
    }

    set_lookup_cost(n){
      $('#timecost').show();
      $('#timecost p').text('Lookup cost: -$' + n.toString());
      this.lookup_cost = n;
    }


    create_route_graph(start,finish,allow_all_edges = false){
      let retval = {};
      let self = this;
      _.forEach(this.is_selected,function(val,key,obj){
        if(val || allow_all_edges){
          var city_from = self.clean_label(self.map.nodes.get(self.map.edges.get(key).from).label)
          var city_to = self.clean_label(self.map.nodes.get(self.map.edges.get(key).to).label)
          var reward = self.get_reward(self.map.nodes.get(self.map.edges.get(key).to).label)
          if(reward==undefined){
            reward = self.get_average_reward(city_to)
          }
          //var reward = self.city_info[city_to]["actual_reward"]
          if (city_from == start){
            city_from = "start"
          }
          if (city_from == finish){
            city_from = "finish"
          }
          if (city_to == start){
            city_to = "start"
          }
          if (city_to == finish){
            city_to = "finish"
          }
          if(retval[city_from]===undefined){
            retval[city_from] = {}
          }
          if(retval[city_to]===undefined){
            retval[city_to] = {}
          }
          retval[city_from][city_to] = reward
        }
      })
      return retval
    }

    get_state_string(){
      s = "0"
      for(var i=1;i<this.city_names.length;i++){
        label = this.map.nodes.get(i).label
        if(label.split('\n').length==1){
          s+="-X"
        }
        else{
          s+="-"+this.get_reward(label)
        }
      }
      return s
    }

    listen_for_spacebar(callback){
      $(document).off("keypress").on( "keypress", function(e){
        if(e.keyCode==32){
          callback()
        }
      })
    }

    simulate_optimal_solution(){
      optimal_actions = this.get_optimal_actions()
      if(optimal_actions.includes(0)){
        $( "#finishedbutton" ).css({"background-color" : "#00ff00"})
        this.listen_for_spacebar(function(){
          $( "#finishedbutton" ).css({"background-color" : "#999999"})
          route = this.get_optimal_route(this.start_city,this.end_cities,true)
          this.show_route(route)
        })
      }
      else{
        _.forEach(optimal_actions,function(a){
          this.map.nodes.update({id: a-1, color: {background: "#00ff00"}})
        })
        this.listen_for_spacebar(function(){
          action = optimal_actions[Math.floor(Math.random() * optimal_actions.length)]
          this.show_reward_on_city(action-1,this.clean_label(this.city_names[action-1]),false)
          this.listen_for_spacebar(function(){
            _.forEach(optimal_actions,function(a){
              this.map.nodes.update({id: a-1, color: {background: "#E5E7E9"}})
            })
            this.listen_for_spacebar(function(){
              this.simulate_optimal_solution()
            })
          })
        })
      }
    }

    get_optimal_actions(){
      Q=optimal_solution[this.get_state_string()]
      m=Infinity
      retval = []
      for(var i=0;i<Q.length;i++){
        if(Q[i]!=null && Q[i]<m)
        m=Q[i]
      }
      for(var i=0;i<Q.length;i++){
        if(Q[i]==m)
        retval.push(i)
      }
      return retval
    }

    get_optimal_route(start,finish,allow_all_edges = false){
      let route_graph = this.create_route_graph(start,finish,allow_all_edges);
      let optimal_route = dijkstra(route_graph);
      for(var i=0;i<optimal_route.path.length;i++){
        if(optimal_route.path[i] == "start"){
          optimal_route.path[i] = start
        }
        if(optimal_route.path[i] == "finish"){
          optimal_route.path[i] = finish
        }
      }
      return optimal_route.path
    }

    show_route(route,feedback=true,is_practice=false,callback){
      //log_data({"event_type": "Show route", "event_info" : {"route" : route}})
      let map = this.map;
      let self = this;

      let node_ids = _.map(route, function(city){
        return self.map.nodes.getIds({
          filter : function(node){
            return (self.clean_label(node.label) ==city) ;
          }
        });
      });
      _.forEach(node_ids, function(id){
        self.map.nodes.update({id : id, color : {border : self.selected_node_color}, borderWidth : self.node_on_route_width})
      });

      let edge_ids = this.map.edges.getIds({
        filter: function(edge){
          var i = route.indexOf(self.clean_label(self.map.nodes.get(edge.from).label))
          var j = route.indexOf(self.clean_label(self.map.nodes.get(edge.to).label))
          return i>=0 && j>=0 && j-i == 1
        }
      })
      _.forEach(edge_ids,function(id){
        self.map.edges.update({id : id, width : self.edge_on_route_width, color: self.selected_edge_color})
      })

      let route_cost = 0;
      let route_cost_expected = 0;
      for(var i=1;i<node_ids.length && i<route.length;i++){
        for(var j=0;j<node_ids[i].length;j++){
          if(this.use_end_city_reward || !this.end_cities.includes(route[i])){
            //expected score
            if(this.is_reward_revelaed(node_ids[i][j])){
              route_cost_expected += this.get_reward(this.map.nodes.get(node_ids[i][j]).label);
            }else{
              let possible_rewards = this.city_info[this.map.nodes.get(node_ids[i][j]).label]["possible_rewards"];
              let sum = 0;
              for(let el of possible_rewards){
                sum += el
              }
              route_cost_expected += sum / possible_rewards.length;
            }

            // reveal
            this.show_reward_on_city(node_ids[i][j],route[i],false);

            //route cost
            route_cost += this.get_reward(this.map.nodes.get(node_ids[i][j]).label);
          }
        }
      }
      let trial_bonus = Math.max((500 - route_cost - this.lookup_cost)/1000,0);
      let feedbacktext = '';
      //log_data({"event_type": "Route scored", "event_info" : {"route": route, "route_cost" : route_cost, "trial_bonus" : trial_bonus, "total_bonus" : this.bonus, "is_practice" : is_practice}})
      if(is_practice){
        feedbacktext="Route cost: $" + route_cost + ", You would have earned: $" + trial_bonus.toFixed(2);
      }
      else{
        BONUS += trial_bonus;
        feedbacktext = "Incurred route cost: $" + route_cost + ", Bonus so far: $" + BONUS.toFixed(2);
      }
      // log
      this.data.path = this.name2id(route);
      this.data.cost_total = route_cost + this.lookup_cost;
      this.data.cost_path = route_cost;
      this.data.cost_expected = route_cost_expected + this.lookup_cost;

      if(feedback){
        this.map.net.setOptions({interaction : {"selectable" : false}});
        $( "#cityinput [type=submit]" ).off("click");
        $( "#cityinput [type=text]").unbind("keydown").unbind("keyup").autocomplete({"lookup":[]}).autocomplete('disable');
        this.show_feedback(feedbacktext);
        callback();
      }
    }

    get_unambiguous_route(start,finish){
      let self = this;
      let map = this.map;

      let retval = [start];
      let edges = _.map(_.pick(this.is_selected,function(val,key,obj){
        return val
      }),function(val,key,obj){
        var city_from = self.clean_label(self.map.nodes.get(self.map.edges.get(key).from).label)
        var city_to = self.clean_label(self.map.nodes.get(self.map.edges.get(key).to).label)
        return [city_from,city_to]
      })

      let city = start;
      while(city != finish){
        let edges_from_city = _.filter(edges,function(e){
          return e[0]==city
        })
        if(edges_from_city.length !=1){
          break;
        }
        else{
          city = edges_from_city[0][1];
        }
        retval.push(city);
      }
      return retval
    }


    //MODES
    disable_edit_mode(){
      this.map.net.off("selectNode").off("selectEdge").off("hoverNode").off("blurNode")
      this.map.net.setOptions({
        nodes: {
          chosen: {
            label: false,
            node: function(values, id, selected, hovering) {
              values.borderWidth = this.node_border_width;
            }
          }
        },
        edges: {
          selectionWidth: 0
        },
        interaction: {
          dragNodes: false,
          dragView: false,
          hover: false,
          selectable : true
        }
      })
      this.map.net.disableEditMode()
    }

    enable_report_mode(){
      this.disable_edit_mode();
      let self = this;

      this.is_selected = _.object(_.map(self.map.edges.getIds(),function(id){return [id, false]}))

      /*this.map.net.on("selectNode",function(e){
        for(var i=0;i<e.nodes.length;i++){
          self.map.nodes.update({id : e.nodes[i], borderWidth: self.node_border_width})
        }
      })*/

      this.map.net.on("click",function(e){
        if(self.data.clicks.length == 0){
          self.show_feedback("Please look up at least one city before choosing the route.",2000, undefined, 'red');

        } else if(self.data.clicks.length > 1 || self.confirmed|| window.confirm("Are you sure to start planning having only one city looked up?")){
          self.confirmed = true;

          for(var i=0;i<e.edges.length;i++){
            var city_from = self.clean_label(self.map.nodes.get(self.map.edges.get(e.edges[i]).from).label)
            var city_to = self.clean_label(self.map.nodes.get(self.map.edges.get(e.edges[i]).to).label)

            self.is_selected[e.edges[i]] = !self.is_selected[e.edges[i]];
          }
          self.update_edge_colors();

        }
      })

      $( "#cityinput [type=submit]" ).off("click").on("click",function(){
        let city_name = $("#cityinput [type=text]").val();
        //log_data({"event_type": "Reveal city price clicked", "event_info" : {"city" : city_name}})
        let matching_node_ids = self.map.nodes.getIds({
          filter: function(node){
            return node.label.toLowerCase() == city_name.toLowerCase() && city_name.toLowerCase()!= self.start_city.toLowerCase();
          }
        })

        //reveal
        if(matching_node_ids.length==1){
          $("#cityinput [type=submit]").prop( "disabled", true )
          $("#cityinput [type=text]").prop( "disabled", true )
          $("#finishedbutton").prop("disabled",true)
          self.show_feedback("Looking up " + city_name, self.reveal_delay, function(){
            $("#cityinput [type=submit]").prop( "disabled", false )
            $("#cityinput [type=text]").prop( "disabled", false )
            $("#finishedbutton").prop("disabled",false)
            $("#cityinput [type=text]").val('')
            self.set_lookup_cost(self.lookup_cost + self.lookup_fee);
            //log_data({"event_type": "Reveal city price", "event_info" : {"city" : city_name}})
            self.show_reward_on_city(matching_node_ids[0],self.clean_label(self.map.nodes.get(matching_node_ids[0]).label),true,true)
          })
        }
        else if(city_name.length>0){
          self.show_feedback(city_name + " does not exist", 1000);
        }
      })
    }

    enable_editing(){
      this.map.net.on("selectNode", function (params) {
        this.map.net.addEdgeMode();
      });

      this.map.net.setOptions({interaction: {dragNodes: true}})

      /** Delete edge by clicking */
      this.map.net.on("selectEdge", function(e) {
        this.map.edges.remove(e.edges);
      });

      /** Highlight connected edges on node hover */
      this.map.net.on("hoverNode", function(evt) {_.each(this.map.net.getConnectedEdges(evt.node), function(e){this.map.edges.update({id: e, width: 2})}); })
      this.map.net.on("blurNode",	function(evt) {_.each(this.map.net.getConnectedEdges(evt.node), function(e){this.map.edges.update({id: e, width: 1})}); })
    }

    move_icon(map, icon_name,node, xoffset=0,yoffset=0){
      var dom_coords = map.net.canvasToDOM({x:node.x+xoffset,y:node.y+yoffset});
      $("#" + icon_name).css({
        "left" : $("#canvas").offset()["left"] + dom_coords.x - $("#" + icon_name).width()/2,
        "top" : $("#canvas").offset()["top"] + dom_coords.y - $("#" + icon_name).height()/2
      })
    }

    draw_car(){
      let start_city = this.start_city;
      let end_cities = this.end_cities;
      let move_icon = this.move_icon;
      let map = this.map;

      let start_nodes = this.map.nodes.get({
        filter : function(node){
          return node.label == start_city;
        }
      });
      let end_nodes = this.map.nodes.get({
        filter : function(node){
          return end_cities.includes(node.label)
        }
      });

      if(start_nodes.length==1){

        for(var i=0;i<this.end_cities.length;i++){
          $("<p></p>").attr('id','star' + (i+1).toString()).addClass('myicon').addClass('plane').append("<i class='fas fa-plane'></i>").insertAfter("#car");
        }
        $(".myicon").show();

        this.map.net.off("afterDrawing").on("afterDrawing", function (ctx) {
          move_icon(map, "car", start_nodes[0]);
          for(var i=0;i<end_cities.length;i++){
            move_icon(map, 'star' + (i+1).toString(),end_nodes[i],30,-15);
          }
        });
        this.map.net.redraw();
      }
      else {
        console.log("None or multiple start nodes")
      }
    }

    draw_graph(){
      // Construct vis.js nodes
      this.map.nodes = new vis.DataSet();

      for (var i = 0; i < this.city_names.length; i++) {
        this.map.nodes.add({
          id: i,
          label: this.city_names[i],
          x: i * 90, y: 10 // fixed node positions (comment to make random)
        });
      }

      this.map.edges = new vis.DataSet();

      // Display initial this.map
      this.map.net = new vis.Network($("#canvas")[0], this.map, this.vis_options);
      this.map.net.storePositions();
    }


    draw_graph_from_json(){
      // Construct vis.js nodes
      let nodes = new vis.DataSet();
      //this.map.nodes = new vis.DataSet();

      _.each(this.graph_data['nodes'], function(node, key, obj) {
        nodes.add({
          id: node["id"],
          label: node["label"],
          x : node["x"]*$("#canvas").width(),
          y : node["y"]*$("#canvas").height()
        });
      });
      this.map.nodes = nodes;


      let edges = new vis.DataSet();
      //this.map.edges = new vis.DataSet();

      _.each(this.graph_data['edges'], function(edge) {
        edges.add({
          from: edge[0],
          to: edge[1],
        });
      });
      this.map.edges = edges;

      // Display initial this.map
      this.map.net = new vis.Network($("#canvas")[0], this.map, this.vis_options);
      this.map.net.storePositions();
      this.map.net.moveTo({scale:1,position:{x:0,y:0}})
    }

    set_autocomplete(mode){
      if(mode){
        $( "#cityinput [type=text]").unbind("keydown").bind("keydown", function( event ) {
          if ( event.keyCode === 9) {
            event.preventDefault();
            var text_pre = $("#cityinput [type=text]").val()
            var suggestions = $( "#cityinput [type=text]").data()["autocomplete"].suggestions;
            if(suggestions.length>0){
              $( "#cityinput [type=text]").val(suggestions[0].value)
            }
            var text_post = $("#cityinput [type=text]").val()
            //log_data({"event_type": "Autocomplete used", "event_info" : {"text_pre" : text_pre, "text_post" : text_post}})
          }
        }).unbind("keyup").bind("keyup", function( event ) {
          //log_data({"event_type": "Keypress in input", "event_info" : {"keycode" : event.keyCode, "text_post" : $("#cityinput [type=text]").val()}})
          if (event.keyCode === 13){
            event.preventDefault();
            $( "#cityinput [type=submit]" ).click()
            return false;
          }
        }).autocomplete({
          lookup: this.city_names,
          lookupLimit : 3,
          minChars : 1,
          lookupFilter : function (suggestion, query, queryLowerCase) {
            return suggestion['value'].toLowerCase().startsWith(queryLowerCase)
          }
        });
      }
      else {
        $( "#cityinput [type=text]").unbind("keyup").bind("keyup", function( event ) {
          //log_data({"event_type": "Keypress in input", "event_info" : {"keycode" : event.keyCode, "text_post" : $("#cityinput [type=text]").val()}})
        })
      }
    }

    resize_canvas(aspect_ratio){
      let map_width = parseInt($("#map").css("width"), 10);
      let map_height = parseInt($("#map").css("height"), 10);
      if(map_width/map_height > aspect_ratio){
        $("#canvas").css({height: map_height, width: aspect_ratio*map_height});
      }
      else{
        $("#canvas").css({height: map_width/aspect_ratio, width: map_width});
      }
      if(!_.isEmpty(this.map.net)){
        this.map.net.redraw();
      }
    }

    enable_resize_correction(aspect_ratio){
      this.resize_canvas(aspect_ratio);
      let self = this;
      $(window).off('resize').on('resize', function(params){
        self.resize_canvas(aspect_ratio);
      });
    }

};
  return plugin;
})();
