import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from obs_utils import setup_observer
import warnings
import yaml
import os

# Suppress warnings
warnings.filterwarnings('ignore')

def create_pdf_report(csv_file, output_pdf):
    print(f"Reading schedule from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        return

    print("Loading configuration...")
    try:
        with open('obs_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        constraints = config.get('constraints', {})
    except Exception as e:
        print(f"Warning: Could not load obs_config.yaml: {e}")
        constraints = {}

    print("Setting up observer...")
    try:
        observer = setup_observer()
    except Exception as e:
        print(f"Error setting up observer: {e}")
        return
    
    # Prepare ReportLab document
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=landscape(A4),
        rightMargin=1*cm, leftMargin=1*cm,
        topMargin=1*cm, bottomMargin=1*cm
    )
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        fontSize=16,
        spaceAfter=12
    )
    
    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        alignment=TA_LEFT
    )

    right_cell_style = ParagraphStyle(
        'RightCellStyle',
        parent=cell_style,
        alignment=TA_RIGHT
    )

    center_cell_style = ParagraphStyle(
        'CenterCellStyle',
        parent=cell_style,
        alignment=TA_CENTER
    )

    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=colors.white,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )

    # --- Page 1: Cover / Visuals ---
    elements.append(Paragraph("Observation Plan Report", title_style))
    elements.append(Spacer(1, 0.5*cm))
    
    try:
        images_row = []
        if os.path.exists('altitude_vs_time.png'):
            # images_row.append([Paragraph("Altitude vs Time", styles['Heading3']), Image('altitude_vs_time.png', width=13.5*cm, height=10*cm, kind='proportional')])
            # Easier to just put the image in the cell
            img1 = Image('altitude_vs_time.png', width=13.5*cm, height=10*cm, kind='proportional')
            images_row.append(img1)
        else:
            images_row.append(Paragraph("Altitude Plot Missing", styles['Normal']))

        if os.path.exists('observation_counts.png'):
            img2 = Image('observation_counts.png', width=13.5*cm, height=10*cm, kind='proportional')
            images_row.append(img2)
        else:
            images_row.append(Paragraph("Counts Plot Missing", styles['Normal']))
            
        if images_row:
            # Create a table for side-by-side layout
            # Page width A4 landscape ~29.7cm. Margins 1+1=2cm. Usable ~27.7cm.
            # Two images of 13.5cm fits.
            t_images = Table([images_row], colWidths=[13.8*cm, 13.8*cm])
            t_images.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ]))
            elements.append(t_images)
            
    except Exception as e:
        print(f"Warning: Could not add images to report: {e}")
        elements.append(Paragraph(f"Error loading images: {e}", styles['Normal']))

    elements.append(PageBreak())
    
    # --- Page 2: Statistics / Placeholder ---
    elements.append(Paragraph("Statistics", title_style))
    elements.append(Spacer(1, 5*cm))
    elements.append(Paragraph("Statistics Page (Placeholder for Page 2)", styles['Normal']))
    elements.append(PageBreak())
    
    # --- Page 3+: Schedule ---
    # Group by night
    if 'night' in df.columns:
        grouped = df.groupby('night')
    else:
        # If 'night' column missing, treat as single group
        grouped = [(1, df)]

    for night, group in grouped:
        # Title for the night
        # Get the date from the first entry (convert UTC to HST date)
        first_time = Time(group.iloc[0]['start_time'])
        # HST is UTC-10
        local_time = first_time - 10 * u.hour
        date_str = local_time.datetime.strftime('%Y-%m-%d')
        
        elements.append(Paragraph(f"Night {night} - {date_str} (HST)", title_style))
        elements.append(Spacer(1, 0.5*cm))
        
        # Table Headers
        headers = [
            Paragraph('Time (HST)', header_style),
            Paragraph('LST', header_style),
            Paragraph('Target', header_style),
            Paragraph('Air', header_style), # Moved Air
            Paragraph('Teff', header_style),
            Paragraph('Rot<br/>(Start)', header_style),
            Paragraph('Rot<br/>(End)', header_style),
            Paragraph('Moon<br/>Sep', header_style),
            Paragraph('Moon<br/>Illum', header_style),
            Paragraph('Moon<br/>Alt', header_style),
            Paragraph('Note', header_style)
        ]
        
        data = [headers]
        bg_commands = []
        warning_color = colors.Color(1, 0.85, 0.85) # Light Red

        print(f"Processing Night {night}...")
        for idx, row in group.iterrows():
            t_start_utc = Time(row['start_time'])
            t_end_utc = Time(row['end_time'])
            
            t_start_hst = t_start_utc - 10 * u.hour
            t_end_hst = t_end_utc - 10 * u.hour
            
            # Format Times
            t_fmt = f"{t_start_hst.datetime.strftime('%H:%M')} - {t_end_hst.datetime.strftime('%H:%M')}"
            
            # LST
            if 'lst' in row and pd.notna(row['lst']):
                lst_str = str(row['lst'])
            else:
                lst_str = "N/A"
            
            # Teff
            if 'teff' in row and pd.notna(row['teff']):
                teff_val = row['teff']
                teff_str = f"{teff_val:.2f}"
            else:
                teff_val = None
                teff_str = "N/A"

            # Moon Stats
            moon_sep_val = None
            if 'moon_sep' in row and pd.notna(row['moon_sep']):
                try:
                    moon_sep_val = float(row['moon_sep'])
                    moon_sep_str = f"{moon_sep_val:.1f}"
                except ValueError:
                     moon_sep_str = str(row['moon_sep'])
            else:
                moon_sep_str = "N/A"

            moon_illum_val = None
            if 'moon_illum' in row and pd.notna(row['moon_illum']):
                moon_illum_val = float(row['moon_illum'])
                moon_illum_str = f"{moon_illum_val:.2f}"
            else:
                moon_illum_str = "N/A"

            moon_alt_val = None
            if 'moon_alt' in row and pd.notna(row['moon_alt']):
                moon_alt_val = float(row['moon_alt'])
                moon_alt_str = f"{moon_alt_val:.1f}"
            else:
                moon_alt_str = "N/A"
            
            # Formats
            rot_start_val = None
            if 'rot_start' in row and pd.notna(row['rot_start']):
                rot_start_val = float(row['rot_start'])
                rot_start_str = f"{rot_start_val:.1f}"
            else:
                 rot_start_str = "N/A"
            
            rot_end_val = None
            if 'rot_end' in row and pd.notna(row['rot_end']):
                rot_end_val = float(row['rot_end'])
                rot_end_str = f"{rot_end_val:.1f}"
            else:
                 rot_end_str = "N/A"

            airmass_val = float(row['airmass'])
            airmass_str = f"{airmass_val:.2f}"
            
            # Current row index in table (header is 0, so first data is 1)
            row_num = len(data)

            # Check Constraints and Highlight
            # 3: Air
            if 'max_airmass' in constraints and airmass_val > constraints['max_airmass']:
                bg_commands.append(('BACKGROUND', (3, row_num), (3, row_num), warning_color))
            
            # 4: Teff
            if teff_val is not None and 'min_teff' in constraints and teff_val < constraints['min_teff']:
                bg_commands.append(('BACKGROUND', (4, row_num), (4, row_num), warning_color))
            
            # 5: Rot Start
            if rot_start_val is not None:
                if 'rotator_min' in constraints and rot_start_val < constraints['rotator_min']:
                    bg_commands.append(('BACKGROUND', (5, row_num), (5, row_num), warning_color))
                if 'rotator_max' in constraints and rot_start_val > constraints['rotator_max']:
                    bg_commands.append(('BACKGROUND', (5, row_num), (5, row_num), warning_color))

            # 6: Rot End
            if rot_end_val is not None:
                if 'rotator_min' in constraints and rot_end_val < constraints['rotator_min']:
                    bg_commands.append(('BACKGROUND', (6, row_num), (6, row_num), warning_color))
                if 'rotator_max' in constraints and rot_end_val > constraints['rotator_max']:
                    bg_commands.append(('BACKGROUND', (6, row_num), (6, row_num), warning_color))

            # 7: Moon Sep
            if moon_sep_val is not None and 'min_moon_sep' in constraints and moon_sep_val < constraints['min_moon_sep']:
                bg_commands.append(('BACKGROUND', (7, row_num), (7, row_num), warning_color))

            # 8: Moon Illum (Updated to max_)
            if moon_illum_val is not None and 'max_moon_ill' in constraints and moon_illum_val > constraints['max_moon_ill']:
                bg_commands.append(('BACKGROUND', (8, row_num), (8, row_num), warning_color))

            # 9: Moon Alt (Updated to max_)
            if moon_alt_val is not None and 'max_moon_alt' in constraints and moon_alt_val > constraints['max_moon_alt']:
                bg_commands.append(('BACKGROUND', (9, row_num), (9, row_num), warning_color))

            # Use Paragraph for Target to allow wrapping
            target_p = Paragraph(str(row['target']), cell_style)
            
            data.append([
                Paragraph(t_fmt, center_cell_style),        # Time (Center)
                Paragraph(lst_str, right_cell_style),       # LST (Right)
                target_p,                                   # Target (Left - default)
                Paragraph(airmass_str, right_cell_style),   # Air (Right)
                Paragraph(teff_str, right_cell_style),      # Teff (Right)
                Paragraph(rot_start_str, right_cell_style), # Rot Start (Right)
                Paragraph(rot_end_str, right_cell_style),   # Rot End (Right)
                Paragraph(moon_sep_str, right_cell_style),  # Moon Sep (Right)
                Paragraph(moon_illum_str, right_cell_style),# Moon Illum (Right)
                Paragraph(moon_alt_str, right_cell_style),  # Moon Alt (Right)
                Paragraph(str(row['note']), cell_style)     # Note (Left)
            ])
            
        # Create Table
        # Define column widths
        # Page width landscape A4 = 29.7cm. Margins L/R = 1cm each. Usable = 27.7cm.
        col_widths = [
            2.8*cm, # Time
            1.7*cm, # LST
            6.5*cm, # Target
            1.2*cm, # Air (Moved)
            1.2*cm, # Teff
            1.8*cm, # Rot Start
            1.8*cm, # Rot End
            1.6*cm, # Moon Sep
            1.6*cm, # Moon Illum
            1.6*cm, # Moon Alt
            2.0*cm  # Note
        ] # Total approx 25.8cm + margins. Fits comfortably.
        
        t = Table(data, colWidths=col_widths, repeatRows=1)
        
        # Style
        base_style = [
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue), # Header background
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),          # All cells middle vertical align
            ('ALIGN', (0,0), (-1,0), 'CENTER'),            # Header row center align
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),    # All grid lines
            ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.whitesmoke]), # Zebra striping
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]
        
        # Combine base style with conditional backgrounds
        t.setStyle(TableStyle(base_style + bg_commands))
        
        elements.append(t)
        elements.append(PageBreak())
        
    print(f"Building PDF: {output_pdf}")
    try:
        doc.build(elements)
        print("Done.")
    except Exception as e:
        print(f"Error building PDF: {e}")

if __name__ == "__main__":
    create_pdf_report('observation_schedule.csv', 'schedule_report.pdf')
