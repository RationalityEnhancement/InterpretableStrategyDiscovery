/**
 * jspsych-button-response
 * Josh de Leeuw
 *
 * plugin for displaying a stimulus and getting a keyboard response
 *
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["video-attention-check"] = (function () {

    var plugin = {};

    plugin.info = {
        name: "video-attention-check",
        description: '',
        parameters: {
            stimulus: {
                type: jsPsych.plugins.parameterType.VIDEO,
                pretty_name: 'Video',
                default: undefined,
                description: 'The video file to play.'
            },
            choices: {
                type: jsPsych.plugins.parameterType.STRING,
                pretty_name: 'Choices',
                default: undefined,
                array: true,
                description: 'The labels for the buttons.'
            },
            button_html: {
                type: jsPsych.plugins.parameterType.STRING,
                pretty_name: 'Button HTML',
                default: '<button class="jspsych-btn">%choice%</button>',
                array: true,
                description: 'The html of the button. Can create own style.'
            },
            prompt: {
                type: jsPsych.plugins.parameterType.STRING,
                pretty_name: 'Prompt',
                default: null,
                description: 'Any content here will be displayed below the buttons.'
            },
            width: {
                type: jsPsych.plugins.parameterType.INT,
                pretty_name: 'Width',
                default: '',
                description: 'The width of the video in pixels.'
            },
            height: {
                type: jsPsych.plugins.parameterType.INT,
                pretty_name: 'Height',
                default: '',
                description: 'The height of the video display in pixels.'
            },
            autoplay: {
                type: jsPsych.plugins.parameterType.BOOL,
                pretty_name: 'Autoplay',
                default: true,
                description: 'If true, the video will begin playing as soon as it has loaded.'
            },
            controls: {
                type: jsPsych.plugins.parameterType.BOOL,
                pretty_name: 'Controls',
                default: false,
                description: 'If true, the subject will be able to pause the video or move the playback to any point in the video.'
            },
            start: {
                type: jsPsych.plugins.parameterType.FLOAT,
                pretty_name: 'Start',
                default: null,
                description: 'Time to start the clip.'
            },
            stop: {
                type: jsPsych.plugins.parameterType.FLOAT,
                pretty_name: 'Stop',
                default: null,
                description: 'Time to stop the clip.'
            },
            margin_vertical: {
                type: jsPsych.plugins.parameterType.STRING,
                pretty_name: 'Margin vertical',
                default: '0px',
                description: 'The vertical margin of the button.'
            },
            margin_horizontal: {
                type: jsPsych.plugins.parameterType.STRING,
                pretty_name: 'Margin horizontal',
                default: '8px',
                description: 'The horizontal margin of the button.'
            },
            number_attention_checks: {
                type: jsPsych.plugins.parameterType.INT,
                pretty_name: 'Attention checks',
                default: '5',
                description: 'The number of attention check'
            },
            time_to_react: {
                type: jsPsych.plugins.parameterType.INT,
                pretty_name: 'Time to react',
                default: '10',
                description: 'The reaction time for attention checks before the video is reloaded'
            },
            response_allowed_while_playing: {
                type: jsPsych.plugins.parameterType.BOOL,
                pretty_name: 'Response allowed while playing',
                default: true,
                description: 'If true, then responses are allowed while the video is playing. ' +
                    'If false, then the video must finish playing before a response is accepted.'
            },
            check_interval: {
                type: jsPsych.plugins.parameterType.INT,
                pretty_name: 'Check interval',
                default: '15',
                description: 'After how many seconds to check the attention'
            }
        }
    }


    plugin.trial = function (display_element, trial) {

        // setup stimulus
        var video_html = '<div>'
        video_html += '<video id="jspsych-video-button-response-stimulus"';

        if (trial.width) {
            video_html += ' width="' + trial.width + '"';
        }
        if (trial.height) {
            video_html += ' height="' + trial.height + '"';
        }
        if (trial.autoplay & (trial.start == null)) {
            // if autoplay is true and the start time is specified, then the video will start automatically
            // via the play() method, rather than the autoplay attribute, to prevent showing the first frame
            video_html += " autoplay ";
        }
        if (trial.controls) {
            video_html += " controls ";
        }
        if (trial.start !== null) {
            // hide video element when page loads if the start time is specified,
            // to prevent the video element from showing the first frame
            video_html += ' style="visibility: hidden;"';
        }
        video_html += ">";

        var video_preload_blob = jsPsych.pluginAPI.getVideoBuffer(trial.stimulus[0]);
        if (!video_preload_blob) {
            for (var i = 0; i < trial.stimulus.length; i++) {
                var file_name = trial.stimulus[i];
                if (file_name.indexOf('?') > -1) {
                    file_name = file_name.substring(0, file_name.indexOf('?'));
                }
                var type = file_name.substr(file_name.lastIndexOf('.') + 1);
                type = type.toLowerCase();
                if (type == "mov") {
                    console.warn('Warning: video-button-response plugin does not reliably support .mov files.')
                }
                video_html += '<source src="' + file_name + '" type="video/' + type + '">';
            }
        }
        video_html += "</video>";
        video_html += "</div>";

        //display buttons
        var buttons = [];
        if (Array.isArray(trial.button_html)) {
            if (trial.button_html.length == trial.choices.length) {
                buttons = trial.button_html;
            } else {
                console.error('Error in video-button-response plugin. The length of the button_html array does not equal the length of the choices array');
            }
        } else {
            for (var i = 0; i < trial.choices.length; i++) {
                buttons.push(trial.button_html);
            }
        }
        video_html += '<div id="jspsych-video-button-response-btngroup">';
        for (var i = 0; i < trial.choices.length; i++) {
            var str = buttons[i].replace(/%choice%/g, trial.choices[i]);
            video_html += '<div class="jspsych-video-button-response-button" style="cursor: pointer; display: inline-block; margin:' + trial.margin_vertical + ' ' + trial.margin_horizontal + '" id="jspsych-video-button-response-button-' + i + '" data-choice="' + i + '">' + str + '</div>';
        }
        video_html += '</div>';

        // add prompt if there is one
        if (trial.prompt !== null) {
            video_html += trial.prompt;
        }

        display_element.innerHTML = video_html;

        var video_element = display_element.querySelector('#jspsych-video-button-response-stimulus');

        var start_time = performance.now();

        var response = {
            rt: null,
            button: null
        };

        // END OF VIDEO
        video_element.onended = function(){
            end_trial();
        }

        // WINDOW ON LOAD
        let attention_check = 0


        disable_buttons();
        setInterval(function () {
            console.log("attention checks", attention_check)
            checkAttention();
        }, trial.check_interval * 1000);
        if (attention_check > trial.number_attention_checks) {
            console.log("ATTENTION!", attention_check);
            end_trial();
        }

        // FUNCTIONS
        function checkAttention() {
            enable_buttons();
            setTimeout(function () {
                restart_video();
            }, trial.time_to_react * 1000)
        }

        function button_response(e){
            var choice = e.currentTarget.getAttribute('data-choice'); // don't use dataset for jsdom compatibility
            after_response(choice);
        }

        function disable_buttons() {
            console.log("disabled button")
            var btns = document.querySelectorAll('.jspsych-video-button-response-button');
            for (var i = 0; i < btns.length; i++) {
                var btn_el = btns[i].querySelector('button');
                if (btn_el) {
                    btn_el.disabled = true;
                }
                btns[i].removeEventListener('click', button_response);
            }
        }

        function restart_video() {
            disable_buttons()
            video_element.currentTime = 0
            attention_check += 1
        }

        function enable_buttons() {
            var btns = document.querySelectorAll('.jspsych-video-button-response-button');
            for (var i = 0; i < btns.length; i++) {
                var btn_el = btns[i].querySelector('button');
                if (btn_el) {
                    btn_el.disabled = false;
                }
                btns[i].addEventListener('click', button_response);
            }
        }

        function after_response(choice) {

            // measure rt
            var end_time = performance.now();
            var rt = end_time - start_time;
            response.button = parseInt(choice);
            response.rt = rt;

            // after a valid response, the stimulus will have the CSS class 'responded'
            // which can be used to provide visual feedback that a response was recorded
            video_element.className += ' responded';

            // disable all the buttons after a response
            disable_buttons();

            if (trial.response_ends_trial) {
                end_trial();
            }
        }

        function end_trial() {

            // kill any remaining setTimeout handlers
            jsPsych.pluginAPI.clearAllTimeouts();

            // stop the video file if it is playing
            // remove any remaining end event handlers
            display_element.querySelector('#jspsych-video-button-response-stimulus').pause();
            display_element.querySelector('#jspsych-video-button-response-stimulus').onended = function () {
            };

            // gather the data to store for the trial
            var trial_data = {
                rt: response.rt,
                stimulus: trial.stimulus,
                response: response.button
            };

            // clear the display
            display_element.innerHTML = '';

            // move on to the next trial
            jsPsych.finishTrial(trial_data);
        }


    };

    return plugin;
})();