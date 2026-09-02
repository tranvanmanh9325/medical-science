selection_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'
    spacing: dp(10)
    padding: dp(20)

    MDLabel:
        text: "SurRol Simulator v2"
        theme_text_color: "Primary"
        font_style: "H6"
        bold: True
        size_hint: 1.0, 0.1

    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "30dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 0.9
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/ecm_track.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "1. Endoscope Fov Control"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Practise ECM control skills"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/needlepick_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "2. Fundamental Actions"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Practise fundamental actions in surgery" 
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/pegtransfer_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "3. Basic Robot Skill Training Tasks"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Pratise positioning and orienting objects"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0

                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        

'''      

