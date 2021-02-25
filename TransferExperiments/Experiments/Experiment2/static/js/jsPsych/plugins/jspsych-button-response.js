/**
 * jspsych-button-response
 * Josh de Leeuw
 *
 * plugin for displaying a stimulus and getting a keyboard response
 *
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["button-response"] = (function() {

  var plugin = {};

 plugin.info = {
   name: "button-response"
 }

 plugin.trial = function(display_element, trial) {

   trial.vid_src = trial.vid_src;
   trial.url_source = trial.url_source || '';
   trial.vid_width = trial.vid_width || 50;
   trial.number_attention_ckecks = trial.number_attention_ckecks || 1;
   trial.time_to_react = trial.time_to_react || 10;


   display_element.html('');
   display_element.append($('<video>', {
     id: 'video',
     src: trial.vid_src,
     type: "video/mp4",
     style: '	display: block; width: auto; 	margin: 0 auto; width: ' + String(trial.vid_width) + '%'
   }));
   display_element.append($('<track>', {
     kind: "subtitles",
     srclang: "en",
     src: 'static/videos/growthMindset.srt'
   }));


      /*
   display_element.html('');
   display_element.append($('<video>', {
     id: 'video-cont'
   }));
   $('#video-cont').append($('<source>', {
     id: 'video',
     src: trial.vid_src,
     type: "video/mp4",
     style: '	display: block; width: auto; 	margin: 0 auto; width: ' + String(trial.vid_width) + '%'
   })).append($('<track>', {
     label: "English",
     kind: "subtitles",
     srclang: "en",
     src: 'static/videos/growthMindset.vtt'
   }))
   */

   display_element.append($('<div>', {
     html: 'Video Source: ' + trial.url_source,
     class: 'text',
     style: 'text-align: center; font-size: 8pt'
   }));
   display_element.append($('<div>', {
     html: "<br> Please watch the video carefully and with audio. <br> Click the attention check button within " + String(trial.time_to_react) + " seconds after it appeared, otherwise the video starts from the beginning. <br> The next page appears automatically when the video finishes.",
     class: 'text',
     style: 'text-align: center;'
   }));
   display_element.append($('<button>', {
     html: "I'm still watching!",
     id: 'attention-button',
     class: 'block-center'
   }).hide().click(function() {
     clearAttention();
   }));


   var trial_data = {
     "restarts": -1
   };
   var attention_interval = null;
   var attention_timer = null;
   var video = $('#video').get(0);


   //start
   video.onloadedmetadata = function() {
     restartVideo();
   }

   // end
   video.onended = function() {
     clearTimeout(attention_timer);
     clearInterval(attention_interval);
     jsPsych.finishTrial(trial_data);
   };

   video.append($('<track>', {
     default: true,
     kind: 'subtitles',
     srclang: 'en',
     label: 'eng',
     src: 'static/videos/growthMindset.vtt'
   }));

   // --------  FUNCTIONS --------
   function checkAttention() {
     $('#attention-button').show();
     console.log(video.muted);
     console.log($("video").prop('muted'));
     attention_timer = setTimeout(function(){ restartVideo()}, trial.time_to_react * 1000);
   }

   function clearAttention(){
     clearTimeout(attention_timer);
     $('#attention-button').hide();
   }

   function restartVideo(){
     //reset
     clearAttention();

     video.pause();
     video.currentTime = 0;
     video.play();
     trial_data.restarts ++;

     // attention checks
     clearInterval(attention_interval);
     if(trial.number_attention_ckecks > 0){
       attention_interval = setInterval(function(){checkAttention()}, 1000 * parseInt(video.duration +1)/(trial.number_attention_ckecks+1));
     }
   }

 };

 return plugin;
})();
