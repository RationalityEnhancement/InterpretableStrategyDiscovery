
jsPsych.plugins['mortgage'] = (function(){

    var plugin = {};

    plugin.info = {
        name: 'mortgage',
        parameters: {
            stimulus: {
                type: jsPsych.plugins.parameterType.HTML_STRING,
                pretty_name: 'Stimulus',
                default: undefined,
                description: 'The HTML string to be displayed'
            },
        plan_buttons: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Plans',
            default: undefined,
            array: true,
            description: 'The available plans'
        },
        interest_rate_buttons: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Interest rates',
            default: undefined,
            array: true,
            description: 'The interest rates for the plans'
        },
        plan_button_html: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Button HTML',
            default: '<button class="jspsych-btn">%plan_buttons%</button>',
            array: true,
            description: 'The html of the button. Can create own style.'
        },
        interest_rate_button_html: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Button HTML',
            default: '<button class="jspsych-btn">%interest_rate_buttons%</button>',
            array: true,
            description: 'The html of the button. Can create own style.'
        },
        prompt: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Prompt',
            default: null,
            description: 'Any content here will be displayed under the button.'
        },
        response_ends_trial: {
            type: jsPsych.plugins.parameterType.BOOL,
            pretty_name: 'Response ends trial',
            default: true,
            description: 'If true, then trial will end when user responds.'
        },
        margin_vertical: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Margin vertical',
            default: '17px',
            description: 'The vertical margin of the button.'
        },
        margin_horizontal: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Margin vertical',
            default: '2px',
            description: 'The vertical margin of the button.'
        },
        button_size_horizontal: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Button size horizontal',
            default: '150px',
            description: 'The horizontal size of the button.'
        },
        button_size_vertical: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Button size horizontal',
            default: '150px',
            description: 'The horizontal size of the button.'
        },
        max_number_of_clicks: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Maximum number of clicks',
            default: 5,
            description: 'The maximum number of clicks.'
        },
        interest_rate_values: {
            type: jsPsych.plugins.parameterType.FLOAT,
            pretty_name: 'Interest rate values',
            default: ['','','','','','','','',''],
            array: true,
            description: 'The values of the interest rates.'
        },
        best_plan: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Best plan',
            default: '',
            description: 'The best plan.'
        },
        plan_values: {
            type: jsPsych.plugins.parameterType.FLOAT,
            pretty_name: 'Plan values',
            default: '',
            description: 'Value to be paid for each plan'
        },
        start_score: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Start score',
            default: 0,
            description: 'The score you start with'
        },
        trial_count: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Trial count',
            default: null,
            description: 'Trial count'
        }
        }
    }

    plugin.trial = function(display_element, trial){

        var html = '<div id="jspsych-html-mortgage-plan-stimulus" style="margin-top:50px; margin-bottom: 40px">'+trial.stimulus+'</div>';

        //var score_html = '<div id="jspsych-html-mortgage-score" style="margin-bottom: 20px">Score: '+SCORE+'</div>';
        //html += score_html;

        // VARIABLE INIT
        let click_counter = 0;
        let click_counter_internal = 0;
        let number_of_clicks_left = trial.max_number_of_clicks

        //create list of plan buttons
        var plan_buttons = [];
        for (var i = 0; i < trial.plan_buttons.length; i++) {
            plan_buttons.push(trial.plan_button_html);
        }

        var plan_str_0 = plan_buttons[0].replace(/%plan_buttons%/g, trial.plan_buttons[0]);
        var plan_str_1 = plan_buttons[1].replace(/%plan_buttons%/g, trial.plan_buttons[1]);
        var plan_str_2 = plan_buttons[2].replace(/%plan_buttons%/g, trial.plan_buttons[2]);


        // display list of interest rate buttons
        var interest_rate_buttons = [];
        for (var i = 0; i < trial.interest_rate_buttons.length; i++) {
            interest_rate_buttons.push(trial.interest_rate_button_html);
        }

        var interest_rate_str_0 = interest_rate_buttons[0].replace(/%interest_rate_buttons%/g, trial.interest_rate_buttons[0]);
        var interest_rate_str_1 = interest_rate_buttons[1].replace(/%interest_rate_buttons%/g, trial.interest_rate_buttons[1]);
        var interest_rate_str_2 = interest_rate_buttons[2].replace(/%interest_rate_buttons%/g, trial.interest_rate_buttons[2]);
        var interest_rate_str_3 = interest_rate_buttons[3].replace(/%interest_rate_buttons%/g, trial.interest_rate_buttons[3]);
        var interest_rate_str_4 = interest_rate_buttons[4].replace(/%interest_rate_buttons%/g, trial.interest_rate_buttons[4]);
        var interest_rate_str_5 = interest_rate_buttons[5].replace(/%interest_rate_buttons%/g, trial.interest_rate_buttons[5]);
        var interest_rate_str_6 = interest_rate_buttons[6].replace(/%interest_rate_buttons%/g, trial.interest_rate_buttons[6]);
        var interest_rate_str_7 = interest_rate_buttons[7].replace(/%interest_rate_buttons%/g, trial.interest_rate_buttons[7]);
        var interest_rate_str_8 = interest_rate_buttons[8].replace(/%interest_rate_buttons%/g, trial.interest_rate_buttons[8]);


        // add instructons
        if(trial.show_instructions){
          html += trial.instructions_html;
        }

        var table_html = '<table style="margin-left:auto;margin-right:auto;margin-bottom: 10px">' +
            '<tr class="mortgage-tablerow">' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-plan-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-plan-button-' + 0 +'" data-choice="Plan A">'+plan_str_0+'</div></th>' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-interest-rate-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-interest-rate-button-' + 0 +'" data-choice="'+0+'">'+interest_rate_str_0+'</div></th>' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-interest-rate-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-interest-rate-button-' + 1 +'" data-choice="'+1+'">'+interest_rate_str_1+'</div></th>' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-interest-rate-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-interest-rate-button-' + 2 +'" data-choice="'+2+'">'+interest_rate_str_2+'</div></th>' +
            '</tr>' +
            '<tr class="mortgage-tablerow">' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-plan-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-plan-button-' + 1 +'" data-choice="'+'Plan B'+'">'+plan_str_1+'</div></th>' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-interest-rate-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-interest-rate-button-' + 3 +'" data-choice="'+3+'">'+interest_rate_str_3+'</div></th>' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-interest-rate-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-interest-rate-button-' + 4 +'" data-choice="'+4+'">'+interest_rate_str_4+'</div></th>' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-interest-rate-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-interest-rate-button-' + 5 +'" data-choice="'+5+'">'+interest_rate_str_5+'</div></th>' +
            '</tr>' +
            '<tr class="mortgage-tablerow">' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-plan-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-plan-button-' + 2 +'" data-choice="'+'Plan C'+'">'+plan_str_2+'</div></th>' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-interest-rate-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-interest-rate-button-' + 6 +'" data-choice="'+6+'">'+interest_rate_str_6+'</div></th>' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-interest-rate-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-interest-rate-button-' + 7 +'" data-choice="'+7+'">'+interest_rate_str_7+'</div></th>' +
            '<th style="text-align: center;"><div class="jspsych-html-mortgage-interest-rate-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-interest-rate-button-' + 8 +'" data-choice="'+8+'">'+interest_rate_str_8+'</div></th>' +
            '</tr>' +
            '</table>'

        html += table_html;

        //show prompt if there is one
        if (trial.prompt !== null) {
            html += trial.prompt;
        }


        // Number of clicks left
        var number_of_clicks_left_html = '<div id="number_of_clicks_left" style="margin-top: 2em;">You can still lookup ' +number_of_clicks_left+ ' interest rates</div>';
        html += number_of_clicks_left_html;


        // Add a button for next trial
        var next_trial_button_html = '<button class="jspsych-btn" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-mortgage-next-trial-button" disabled>Next</button>';

        html += next_trial_button_html


        display_element.innerHTML = html;

        // clicking on interest rate button
        for (let i = 0; i < trial.interest_rate_buttons.length; i++){

            let button = display_element.querySelector('#jspsych-html-mortgage-interest-rate-button-' + i).childNodes[0]
            let periods = ['2022', '2023 – 2027', '2028 – 2052']
            button.innerHTML =  periods[i % 3] + '<br> ?';

            display_element.querySelector('#jspsych-html-mortgage-interest-rate-button-' + i).addEventListener('click', function(e){

              // double check
              if(!button.disabled){

                // console.log("clicked interest button")

                // Add content to button
                button.innerHTML =  periods[i % 3] + '<br> <span onclick="return false;" style="font-weight:bold; font-size: 1.2em">' + trial.interest_rate_values[i] + '</span>';

                // disable button
                button.setAttribute('disabled', 'disabled');
                button.classList.add('disabled');
                button.classList.add('clicked');

                // log click
                let interest_rate_choice = e.currentTarget.getAttribute('data-choice');
                if (click_counter < trial.max_number_of_clicks){
                    click_counter += 1;
                }
                click_counter_internal += 1;
                after_response_interest_rate_button(click_counter, interest_rate_choice);
                update_click_counter(click_counter)
            }
        })}

        // clicking on plan button
        for (let i = 0; i < trial.plan_buttons.length; i++){
            let button = display_element.querySelector('#jspsych-html-mortgage-plan-button-' + i);
            button.addEventListener('click', function(e){

                // disable plan buttons
                var btn_dis = document.querySelectorAll('.jspsych-html-mortgage-plan-button button');
                for(var i=0; i<btn_dis.length; i++){
                    btn_dis[i].setAttribute('disabled', 'disabled');
                    btn_dis[i].classList.add('disabled');
                }

                // highlight clicked one
                button.childNodes[0].classList.add('clicked');

                // disable mortgage buttons
                update_click_counter(3)


                //update the prompt text
                /*
                if (plan_choice == trial.best_plan){
                    var text_str = 'You have selected the cheapest plan! Your score increased by +10 points. ';
                    document.querySelector('#mortgage-prompt').textContent = text_str;
                } else {
                    var text_str = 'You have not selected the cheapest plan. Your score decreased by -10 points. ';
                    document.querySelector('#mortgage-prompt').textContent = text_str;
                }*/
                let plan_choice = e.currentTarget.getAttribute('data-choice');
                document.querySelector('#number_of_clicks_left').textContent = 'You have selected mortgage ' + plan_choice + '. Please proceed to the next trial.'
                after_response_plan_button(plan_choice);
            })}


        // start time
        var start_time = performance.now();

        // store response
        var response = {
            rt: null,
            plan_button: null,
            interest_rate_button: [],
            score: null,
            trial_score: null,
            all_node_values: []
        };

        // FUNCTION
        function update_click_counter(click_counter){
            number_of_clicks_left = trial.max_number_of_clicks - click_counter
            document.querySelector('#number_of_clicks_left').textContent = "You can still lookup " + number_of_clicks_left + " interest rates";
            if (number_of_clicks_left == 0) {
                // disable all the buttons after a response
                var btns = document.querySelectorAll('.jspsych-html-mortgage-interest-rate-button button');
                for(var i=0; i<btns.length; i++){
                    //btns[i].removeEventListener('click');
                    btns[i].setAttribute('disabled', 'disabled');
                    btns[i].classList.add('disabled');
                }
            }
        }

        // function to handle responses by the subject
        function after_response_interest_rate_button(click_counter, interest_rate_choice) {
            // measure rt
            var end_time = performance.now();
            var rt = end_time - start_time;
            response.interest_rate_button.push(parseInt(interest_rate_choice));
            response.rt = rt;
            response.score = SCORE;
            response.all_node_values = trial.interest_rate_values;

            // after a valid response, the stimulus will have the CSS class 'responded'
            // which can be used to provide visual feedback that a response was recorded
            // display_element.querySelector('#jspsych-html-mortgage-stimulus').className += ' responded';

            if (click_counter_internal > trial.max_number_of_clicks) {
                // number_of_clicks_left = 3
                // document.querySelector('#number_of_clicks_left').textContent = number_of_clicks_left;
                end_trial();
            }
        }

        function after_response_plan_button(plan_choice) {
            // measure rt
            var end_time = performance.now();
            var rt = end_time - start_time;
            var trial_score = 0;
            response.plan_button = plan_choice;
            response.rt = rt;

            // if clicked button value = best plan, then bonus +10, otherwise bonus -10
            if (plan_choice == trial.best_plan){
                SCORE += 10;
                // console.log("selected best plan", SCORE);
                //document.querySelector('#jspsych-html-mortgage-score').textContent = "Score: " + SCORE;
                trial_score = 10
            } else {
                SCORE -= 10;
                // console.log("did not select the best plan", SCORE);
                //document.querySelector('#jspsych-html-mortgage-score').textContent = "Score: " + SCORE;
                trial_score = -10
            }
            response.score = SCORE;
            response.trial_score = trial_score

            // enable next trial button
            var next_trial_butt = document.querySelector('#jspsych-html-mortgage-next-trial-button');
            next_trial_butt.disabled = false;
            display_element.querySelector('#jspsych-html-mortgage-next-trial-button').addEventListener('click', function(e){
                end_trial();
            })

            // disable all o
        }



            // function to end trial when it is time
        function end_trial() {
            // kill any remaining setTimeout handlers
            jsPsych.pluginAPI.clearAllTimeouts();

            // gather the data to store for the trial
            var trial_data = {
                rt: response.rt,
                stimulus: trial.stimulus,
                response_plan: response.plan_button,
                response_interest_rate: response.interest_rate_button,
                score: response.score,
                trial_score: response.trial_score,
                all_node_values: response.all_node_values
            };
            // clear the display
            display_element.innerHTML = '';
            // move on to the next trial
            // console.log(trial_data)
            jsPsych.finishTrial(trial_data);
        }

    };

    return plugin;
})();
