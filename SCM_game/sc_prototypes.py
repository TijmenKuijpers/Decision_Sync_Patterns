from simpn.prototypes import Prototype
import inspect
import simpn.visualisation as vis
import pygame
from simpn.simulator import SimToken, SimVar


COLOR_SC_RED = (254, 61, 82)
COLOR_SC_GREEN = (92, 220, 110)
COLOR_SC_YELLOW = (229, 217, 71)
COLOR_SC_BLUE = (55, 155, 218)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class SCOrder(Prototype):
    def __init__(self, model, incoming, outgoing, name, behavior, guard=None, outgoing_behavior=None):
        super().__init__(model, incoming, outgoing, name)

        busyvar_name = name + "_busy"
        start_event_name = name + "<task:start>"
        complete_event_name = name + "<task:complete>"
        self._busyvar = model.add_var(busyvar_name)
        self.add_var(self._busyvar)
        start_event = model.add_event(incoming, [self._busyvar], behavior, name=start_event_name, guard=guard)
        self.add_event(start_event)
        if outgoing_behavior is None:
            complete_event = model.add_event([self._busyvar], outgoing, lambda b: [SimToken(b[i]) for i in range(len(b))] if b is tuple else [SimToken(b)], name=complete_event_name)
        else:
            complete_event = model.add_event([self._busyvar], outgoing, outgoing_behavior, name=complete_event_name)
        self.add_event(complete_event)

        model.add_prototype(self)

    class SCOrderViz(vis.Node):
        def __init__(self, model_node):
            super().__init__(model_node)
            self._width = 100
            self._height = vis.STANDARD_NODE_HEIGHT
            self._half_width =  self._width / 2
            self._half_height = self._height / 2

        def draw(self, screen):
            x_pos, y_pos = int(self._pos[0] - self._width/2), int(self._pos[1] - self._height/2)
            pygame.draw.rect(screen, COLOR_SC_BLUE, pygame.Rect(x_pos, y_pos, self._width, self._height), border_radius=int(0.075*self._width))
            pygame.draw.rect(screen, COLOR_SC_BLUE, pygame.Rect(x_pos, y_pos, self._width, self._height),  vis.LINE_WIDTH, int(0.075*self._width))
            font = pygame.font.SysFont('Calibri', vis.TEXT_SIZE)
            bold_font = pygame.font.SysFont('Calibri', vis.TEXT_SIZE, bold=True)

            # draw label, use a white font color
            label = bold_font.render(self._model_node.get_id(), True, WHITE)
            text_x_pos = int((self._width - label.get_width())/2) + x_pos
            text_y_pos = int((self._height - label.get_height())/2) + y_pos
            screen.blit(label, (text_x_pos, text_y_pos))

            # draw marking
            mstr = "["
            ti = 0
            for token in self._model_node._busyvar.marking:
                mstr += str(token.value) + "@" + str(round(token.time, 2))
                if ti < len(self._model_node._busyvar.marking) - 1:
                    mstr += ", "
                ti += 1
            mstr += "]"
            label = bold_font.render(mstr, True, BLACK)
            text_x_pos = self._pos[0] - int(label.get_width()/2)
            text_y_pos = self._pos[1] + self._half_height + vis.LINE_WIDTH
            screen.blit(label, (text_x_pos, text_y_pos))        

    def get_visualisation(self):
        return self.SCOrderViz(self)


class SCDemand(SimVar):
    def __init__(self, model, _id, priority=lambda token: token.time):
        super().__init__(_id, priority)

        model.add_prototype_var(self)

    class SCDemandViz(vis.Node):
        def __init__(self, model_node):
            super().__init__(model_node)
        
        def draw(self, screen):
            pygame.draw.circle(screen, COLOR_SC_GREEN, (self._pos[0], self._pos[1]), self._width/2)
            pygame.draw.circle(screen, COLOR_SC_GREEN, (self._pos[0], self._pos[1]), self._width/2, vis.LINE_WIDTH)    
            font = pygame.font.SysFont('Calibri', vis.TEXT_SIZE)
            bold_font = pygame.font.SysFont('Calibri', vis.TEXT_SIZE, bold=True)

            # draw label
            label = font.render(self._model_node.get_id(), True, BLACK)
            text_x_pos = self._pos[0] - int(label.get_width()/2)
            text_y_pos = self._pos[1] + self._half_height + vis.LINE_WIDTH
            screen.blit(label, (text_x_pos, text_y_pos))

            # draw marking
            mstr = "["
            ti = 0
            for token in self._model_node.marking:
                mstr += str(token.value) + "@" + str(round(token.time, 2))
                if ti < len(self._model_node.marking) - 1:
                    mstr += ", "
                ti += 1
            mstr += "]"
            label = bold_font.render(mstr, True, BLACK)
            text_x_pos = self._pos[0] - int(label.get_width()/2)
            text_y_pos = self._pos[1] + self._half_height + vis.LINE_WIDTH + int(label.get_height())
            screen.blit(label, (text_x_pos, text_y_pos))        

    def get_visualisation(self):
        return self.SCDemandViz(self)


class SCStock(SimVar):
    def __init__(self, model, _id, priority=lambda token: token.time):
        super().__init__(_id, priority)

        model.add_prototype_var(self)

    class SCStockViz(vis.Node):
        def __init__(self, model_node):
            super().__init__(model_node)
        
        def draw(self, screen):
            x, y = self._pos
            hw, hh = self._half_width, self._half_height
            pygame.draw.polygon(screen, COLOR_SC_YELLOW, [(x-hw, y-hh), (x+hw, y-hh), (x, y+hh), (x-hw, y-hh)])
            pygame.draw.polygon(screen, COLOR_SC_YELLOW, [(x-hw, y-hh), (x+hw, y-hh), (x, y+hh), (x-hw, y-hh)], vis.LINE_WIDTH)
            font = pygame.font.SysFont('Calibri', vis.TEXT_SIZE)
            bold_font = pygame.font.SysFont('Calibri', vis.TEXT_SIZE, bold=True)

            # draw label
            label = font.render(self._model_node.get_id(), True, BLACK)
            text_x_pos = self._pos[0] - int(label.get_width()/2)
            text_y_pos = self._pos[1] + self._half_height + vis.LINE_WIDTH
            screen.blit(label, (text_x_pos, text_y_pos))

            # draw marking
            mstr = "["
            ti = 0
            for token in self._model_node.marking:
                mstr += str(token.value) + "@" + str(round(token.time, 2))
                if ti < len(self._model_node.marking) - 1:
                    mstr += ", "
                ti += 1
            mstr += "]"
            label = bold_font.render(mstr, True, BLACK)
            text_x_pos = self._pos[0] - int(label.get_width()/2)
            text_y_pos = self._pos[1] + self._half_height + vis.LINE_WIDTH + int(label.get_height())
            screen.blit(label, (text_x_pos, text_y_pos))        

    def get_visualisation(self):
        return self.SCStockViz(self)


class SCSchedule(SimVar):
    def __init__(self, model, _id, priority=lambda token: token.time):
        super().__init__(_id, priority)

        model.add_prototype_var(self)

    class SCScheduleViz(vis.Node):
        def __init__(self, model_node):
            super().__init__(model_node)
        
        def draw(self, screen):
            if len(self._model_node.marking) == 0:
                return
            
            bold_font = pygame.font.SysFont('Calibri', vis.TEXT_SIZE, bold=True)

            # draw schedule as table
            # each token contains a tuple (order_id, steps). Each step is a tuple (step_name, time).
            # the step names are the same for each tuple.
            # we draw the schedule as a table with the order_id as the row header and the steps as the columns

            # get the step names
            step_names = [step[0] for step in self._model_node.marking[0].value[1:]]
            # prepend with order_id
            step_names.insert(0, "Order ID")

            x_pos, y_pos = int(self._pos[0] - self._half_width), int(self._pos[1] - self._half_height)
            cell_widths = [bold_font.render(step_name, True, BLACK).get_width()+10 for step_name in step_names]
            cell_height = bold_font.render("0", True, BLACK).get_height() + 10

            # draw the header at -half_width, -half_height
            for i, step_name in enumerate(step_names):
                pygame.draw.rect(screen, BLACK, pygame.Rect(x_pos, y_pos, cell_widths[i], cell_height))
                pygame.draw.rect(screen, BLACK, pygame.Rect(x_pos, y_pos, cell_widths[i], cell_height), vis.LINE_WIDTH)
                label = bold_font.render(step_name, True, WHITE)
                text_x_pos = x_pos + 5
                text_y_pos = y_pos + 5
                screen.blit(label, (text_x_pos, text_y_pos))
                x_pos += cell_widths[i]
            # draw the rows below that
            for i, token in enumerate(self._model_node.marking):
                x_pos = int(self._pos[0] - self._half_width)
                y_pos = int(self._pos[1] - self._half_height + (i+1)*cell_height)
                for j, step in enumerate(token.value):
                    if j > 0 and step[2]:
                        pygame.draw.rect(screen, COLOR_SC_RED, pygame.Rect(x_pos, y_pos, cell_widths[j], cell_height))
                    pygame.draw.rect(screen, BLACK, pygame.Rect(x_pos, y_pos, cell_widths[j], cell_height), vis.LINE_WIDTH)
                    if j == 0:
                        label = bold_font.render(str(step), True, BLACK)
                    else:
                        label = bold_font.render(str(step[1]), True, BLACK)
                    text_x_pos = x_pos + 5
                    text_y_pos = y_pos + 5
                    screen.blit(label, (text_x_pos, text_y_pos))
                    x_pos += cell_widths[j]


    def get_visualisation(self):
        return self.SCScheduleViz(self)
